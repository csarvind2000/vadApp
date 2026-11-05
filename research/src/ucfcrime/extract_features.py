"""
Snippet-level feature extraction for UCF-Crime.

Decodes each video once, groups frames into non-overlapping snippets of
``snippet_len`` frames, and runs a Kinetics-pretrained 3D backbone over them.
Output per video is a float32 array of shape ``[n_snippets, n_crops, D]`` saved
as ``<stem>.npy``, plus a metadata record used to build ``video_meta.json``.

Backbones
---------
``i3d_r50``      pytorchvideo I3D ResNet50, Kinetics-400, D=2048
``r2plus1d_18``  torchvision R(2+1)D-18,    Kinetics-400, D=512
``swin3d_t``     torchvision Video Swin-T,  Kinetics-400, D=768
``s3d``          torchvision S3D,           Kinetics-400, D=1024

Note on comparability: most published UCF-Crime numbers use I3D-*Inception*
(D=1024) from the original Carreira & Zisserman release. ``i3d_r50`` here is
I3D-*ResNet50* (D=2048). Because train and test features are extracted with the
same backbone, results are internally consistent, but a reproduction of Sultani
et al. should be judged against the published *range* rather than 75.41 exactly.

Cropping
--------
``--n_crops 1``   centre crop -- fast, use for development and ablations
``--n_crops 10``  4 corners + centre, each also flipped -- the convention behind
                  most published numbers; ~10x the GPU cost, typically worth
                  about one AUC point.

Usage
-----
    python -m src.ucfcrime.extract_features \\
        --videos_root /media/.../UCF-Crime/Videos \\
        --out_dir /media/.../UCF-Crime/features/i3d_r50_c1 \\
        --backbone i3d_r50 --n_crops 1 --batch_size 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from .splits import load_test_split, load_train_split

# Kinetics normalisation used by both pytorchvideo and torchvision video models.
MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD = np.array([0.225, 0.225, 0.225], dtype=np.float32)
_MEAN_T = torch.tensor(MEAN)
_STD_T = torch.tensor(STD)

RESIZE_SHORT = 256
CROP = 224
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
def build_backbone(name: str, device: torch.device) -> Tuple[nn.Module, int]:
    """Return a feature extractor mapping [B,3,T,H,W] -> [B,D], and D."""
    if name == "i3d_r50":
        from pytorchvideo.models.hub import i3d_r50
        net = i3d_r50(pretrained=True)
        blocks = nn.Sequential(*list(net.blocks[:-1]))   # drop classification head

        class I3DFeat(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.b = b

            def forward(self, x):
                return torch.nn.functional.adaptive_avg_pool3d(self.b(x), 1).flatten(1)

        return I3DFeat(blocks).to(device).eval(), 2048

    import torchvision.models.video as tvv
    spec = {
        "r2plus1d_18": (tvv.r2plus1d_18, tvv.R2Plus1D_18_Weights.KINETICS400_V1, 512),
        "r3d_18": (tvv.r3d_18, tvv.R3D_18_Weights.KINETICS400_V1, 512),
        "mc3_18": (tvv.mc3_18, tvv.MC3_18_Weights.KINETICS400_V1, 512),
        "s3d": (tvv.s3d, tvv.S3D_Weights.KINETICS400_V1, 1024),
        "swin3d_t": (tvv.swin3d_t, tvv.Swin3D_T_Weights.KINETICS400_V1, 768),
        "swin3d_s": (tvv.swin3d_s, tvv.Swin3D_S_Weights.KINETICS400_V1, 768),
    }
    if name not in spec:
        raise ValueError(f"unknown backbone {name!r}; choose from "
                         f"{['i3d_r50'] + sorted(spec)}")
    ctor, weights, dim = spec[name]
    net = ctor(weights=weights)
    # Replace the classifier with identity so forward() returns pooled features.
    if hasattr(net, "fc"):
        net.fc = nn.Identity()
    elif hasattr(net, "head"):
        net.head = nn.Identity()
    elif hasattr(net, "classifier"):
        net.classifier = nn.Identity()
    return net.to(device).eval(), dim


# ---------------------------------------------------------------------------
# Decoding and cropping
# ---------------------------------------------------------------------------
def _resize_short(frame: np.ndarray, short: int = RESIZE_SHORT) -> np.ndarray:
    h, w = frame.shape[:2]
    if min(h, w) == short:
        return frame
    scale = short / min(h, w)
    return cv2.resize(frame, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_LINEAR)


def _crops(clip: np.ndarray, n_crops: int) -> np.ndarray:
    """
    clip: [T,H,W,3] float32 in [0,1]. Returns [n_crops,T,CROP,CROP,3].

    n_crops=1  -> centre crop
    n_crops=10 -> 4 corners + centre, each plus its horizontal flip
    """
    T, H, W, _ = clip.shape
    cy, cx = (H - CROP) // 2, (W - CROP) // 2
    if n_crops == 1:
        return clip[None, :, cy:cy + CROP, cx:cx + CROP]
    if n_crops != 10:
        raise ValueError("n_crops must be 1 or 10")
    offsets = [(0, 0), (0, W - CROP), (H - CROP, 0), (H - CROP, W - CROP), (cy, cx)]
    out = []
    for y, x in offsets:
        c = clip[:, y:y + CROP, x:x + CROP]
        out.append(c)
        out.append(c[:, :, ::-1])          # horizontal flip
    return np.stack(out, axis=0)


def iter_snippets(path: str, snippet_len: int) -> Iterator[np.ndarray]:
    """Yield [snippet_len,H,W,3] float32 clips in [0,1]; trailing frames dropped."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    buf: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _resize_short(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buf.append(frame.astype(np.float32) / 255.0)
            if len(buf) == snippet_len:
                yield np.stack(buf, axis=0)
                buf = []
    finally:
        cap.release()


def probe(path: str) -> Dict[str, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    eff = fps if fps and fps > 1e-3 else 30.0
    dur = n / eff if n > 0 else 0.0
    return {"fps": fps, "n_frames_decoded": n, "duration_seconds": dur,
            "n_frames_nominal": int(round(dur * 30.0)), "width": w, "height": h}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_video(
    path: str,
    model: nn.Module,
    dim: int,
    device: torch.device,
    snippet_len: int = 16,
    n_crops: int = 1,
    batch_size: int = 8,
    amp: bool = True,
) -> Tuple[np.ndarray, int]:
    """Return ([n_snippets, n_crops, D] float32, n_frames_consumed)."""
    feats: List[np.ndarray] = []
    pending: List[np.ndarray] = []
    n_frames = 0

    def flush():
        if not pending:
            return
        # [B,n_crops,T,H,W,3] -> [B*n_crops,3,T,H,W]
        arr = np.stack(pending, axis=0)
        b, c = arr.shape[0], arr.shape[1]
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(device, non_blocking=True)
        # Normalise on the GPU: the CPU is saturated decoding, so keep work off it.
        t = (t - _MEAN_T.to(t.device)) / _STD_T.to(t.device)
        t = t.permute(0, 1, 5, 2, 3, 4).reshape(b * c, 3, snippet_len, CROP, CROP)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = model(t)
        feats.append(out.float().reshape(b, c, -1).cpu().numpy())
        pending.clear()

    for clip in iter_snippets(path, snippet_len):
        n_frames += snippet_len
        pending.append(_crops(clip, n_crops))
        if len(pending) >= batch_size:
            flush()
    flush()

    if not feats:
        return np.zeros((0, n_crops, dim), dtype=np.float32), 0
    return np.concatenate(feats, axis=0).astype(np.float32), n_frames


def find_videos(videos_root: str) -> Dict[str, str]:
    """Map video stem -> path, searching recursively."""
    out: Dict[str, str] = {}
    for dirpath, _dirs, files in os.walk(videos_root):
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                out.setdefault(os.path.splitext(f)[0], os.path.join(dirpath, f))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--backbone", default="i3d_r50")
    ap.add_argument("--snippet_len", type=int, default=16)
    ap.add_argument("--n_crops", type=int, default=1, choices=[1, 10])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--split", default="all", choices=["all", "train", "test"])
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N videos")
    ap.add_argument("--shard", type=int, default=0, help="this process's shard index")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="split the video list across N processes (decode is CPU-bound)")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dim = build_backbone(args.backbone, device)
    os.makedirs(args.out_dir, exist_ok=True)

    wanted: List[str] = []
    if args.split in ("all", "train"):
        wanted += [v.stem for v in load_train_split()]
    if args.split in ("all", "test"):
        wanted += [v.stem for v in load_test_split()]
    wanted = sorted(dict.fromkeys(wanted))

    available = find_videos(args.videos_root)
    todo = [s for s in wanted if s in available]
    missing = [s for s in wanted if s not in available]
    if args.limit:
        todo = todo[:args.limit]
    if args.num_shards > 1:
        if not 0 <= args.shard < args.num_shards:
            raise SystemExit(f"--shard must be in [0,{args.num_shards})")
        todo = todo[args.shard::args.num_shards]

    print(f"backbone={args.backbone} dim={dim} device={device} crops={args.n_crops}")
    shard_note = "" if args.num_shards == 1 else f" | shard {args.shard}/{args.num_shards}"
    print(f"target videos: {len(wanted)} | this process: {len(todo)} | missing: {len(missing)}{shard_note}")
    if missing:
        print(f"  missing e.g. {missing[:5]}")

    suffix = "" if args.num_shards == 1 else f"_shard{args.shard}"
    meta_path = os.path.join(args.out_dir, f"extract_meta{suffix}.json")
    meta: Dict[str, Dict] = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    t_start = time.time()
    done = skipped = failed = 0
    for i, stem in enumerate(todo, 1):
        out_npy = os.path.join(args.out_dir, stem + ".npy")
        if os.path.isfile(out_npy) and not args.overwrite:
            skipped += 1
            continue
        try:
            info = probe(available[stem])
            feat, n_used = extract_video(
                available[stem], model, dim, device,
                snippet_len=args.snippet_len, n_crops=args.n_crops,
                batch_size=args.batch_size, amp=not args.no_amp,
            )
            if feat.shape[0] == 0:
                raise RuntimeError("no complete snippet decoded")
            np.save(out_npy, feat)
            info.update({"n_snippets": int(feat.shape[0]), "n_crops": int(feat.shape[1]),
                         "dim": int(feat.shape[2]), "n_frames_used": int(n_used)})
            meta[stem] = info
            done += 1
        except Exception as e:  # keep going; report at the end
            failed += 1
            print(f"  [FAIL] {stem}: {type(e).__name__}: {e}")

        if i % 25 == 0 or i == len(todo):
            el = time.time() - t_start
            rate = (done / el) if el > 0 and done else 0.0
            left = (len(todo) - i) / rate if rate > 0 else float("nan")
            print(f"  [{i}/{len(todo)}] done={done} skip={skipped} fail={failed} "
                  f"| {rate:.2f} vid/s | ETA {left/60:.1f} min", flush=True)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\ndone={done} skipped={skipped} failed={failed} -> {args.out_dir}")
    print(f"metadata: {meta_path}")


if __name__ == "__main__":
    main()
