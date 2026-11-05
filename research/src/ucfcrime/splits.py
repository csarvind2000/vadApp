"""
Official UCF-Crime split handling and frame-level ground truth construction.

Source metadata (data/ucfcrime_meta/):
  Anomaly_Train.txt                  1610 lines  -> "Category/Video.mp4"
  Temporal_Anomaly_Annotation.txt     290 lines  -> "Video.mp4  Category  s1 e1 s2 e2"

Protocol reproduced from the authors' Evaluate_Anomaly_Detector.m:
  * Annotation frame indices are 1-based and INCLUSIVE on both ends.
      MATLAB:  st = max(Ann(k,1), 1);  en = min(Ann(k,2), n_frames);  GT(st:en) = 1
      Python:  gt[st - 1 : en] = 1
  * A value of -1 marks an unused interval slot.
  * Normal test videos have all-zero ground truth.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

# The 13 anomaly categories of the benchmark, plus the normal class.
ANOMALY_CLASSES: Tuple[str, ...] = (
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
)
NORMAL_CLASS = "Normal"

# Normal videos live under this directory in the training list, and are labelled
# "Normal" in the test annotation file.
TRAIN_NORMAL_DIR = "Training_Normal_Videos_Anomaly"

DEFAULT_META_DIR = os.path.join("data", "ucfcrime_meta")
TRAIN_LIST = "Anomaly_Train.txt"
TEST_ANN = "Temporal_Anomaly_Annotation.txt"

# Expected counts of the official split -- used as hard assertions so a corrupted
# or substituted metadata file cannot silently change the benchmark.
N_TRAIN, N_TRAIN_ANOM, N_TRAIN_NORMAL = 1610, 810, 800
N_TEST, N_TEST_ANOM, N_TEST_NORMAL = 290, 140, 150


@dataclass(frozen=True)
class TrainVideo:
    """A training video. Only the video-level label is available (weak supervision)."""
    name: str            # "Abuse001_x264.mp4"
    category: str        # "Abuse" or "Normal"
    rel_path: str        # "Abuse/Abuse001_x264.mp4" -- path within the dataset root

    @property
    def is_anomalous(self) -> bool:
        return self.category != NORMAL_CLASS

    @property
    def label(self) -> int:
        return int(self.is_anomalous)

    @property
    def stem(self) -> str:
        return os.path.splitext(self.name)[0]


@dataclass(frozen=True)
class TestVideo:
    """A test video with frame-level temporal annotation."""
    name: str                                        # "Abuse028_x264.mp4"
    category: str                                    # "Abuse" or "Normal"
    intervals: Tuple[Tuple[int, int], ...] = field(default=())  # 1-based inclusive

    @property
    def is_normal(self) -> bool:
        return self.category == NORMAL_CLASS

    @property
    def stem(self) -> str:
        return os.path.splitext(self.name)[0]

    def frame_labels(self, n_frames: int) -> np.ndarray:
        """
        Binary frame-level ground truth of length ``n_frames``.

        Follows the reference MATLAB exactly: 1-based inclusive intervals, start
        clamped to >= 1, end clamped to <= n_frames. Normal videos are all zeros.
        """
        if n_frames <= 0:
            raise ValueError(f"{self.name}: n_frames must be positive, got {n_frames}")
        gt = np.zeros(n_frames, dtype=np.int8)
        if self.is_normal:
            return gt
        for start, end in self.intervals:
            st = max(int(start), 1)
            en = min(int(end), n_frames)
            if en >= st:
                gt[st - 1:en] = 1
        return gt


def _meta_path(meta_dir: str, fname: str) -> str:
    p = os.path.join(meta_dir, fname)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"Missing UCF-Crime metadata file: {p}\n"
            f"Download it from the authors' repository:\n"
            f"  https://github.com/WaqasSultani/AnomalyDetectionCVPR2018/blob/master/{fname}"
        )
    return p


def load_train_split(meta_dir: str = DEFAULT_META_DIR, strict: bool = True) -> List[TrainVideo]:
    """Parse Anomaly_Train.txt into 1610 TrainVideo records."""
    out: List[TrainVideo] = []
    with open(_meta_path(meta_dir, TRAIN_LIST)) as f:
        for raw in f:
            line = raw.strip().replace("\\", "/")
            if not line:
                continue
            if "/" not in line:
                raise ValueError(f"Malformed train entry (expected 'Category/Video.mp4'): {line!r}")
            folder, name = line.split("/", 1)
            category = NORMAL_CLASS if folder == TRAIN_NORMAL_DIR else folder
            if category != NORMAL_CLASS and category not in ANOMALY_CLASSES:
                raise ValueError(f"Unknown training category {category!r} in line {line!r}")
            out.append(TrainVideo(name=name, category=category, rel_path=line))

    if strict:
        n_anom = sum(v.is_anomalous for v in out)
        if (len(out), n_anom) != (N_TRAIN, N_TRAIN_ANOM):
            raise ValueError(
                f"Train split does not match the official protocol: got {len(out)} videos "
                f"({n_anom} anomalous), expected {N_TRAIN} ({N_TRAIN_ANOM} anomalous). "
                f"Do not report results from a modified split."
            )
    return out


def load_test_split(meta_dir: str = DEFAULT_META_DIR, strict: bool = True) -> List[TestVideo]:
    """Parse Temporal_Anomaly_Annotation.txt into 290 TestVideo records."""
    out: List[TestVideo] = []
    with open(_meta_path(meta_dir, TEST_ANN)) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 4:
                raise ValueError(f"Malformed test annotation (need >=4 fields): {line!r}")
            name, category = parts[0], parts[1]
            if category != NORMAL_CLASS and category not in ANOMALY_CLASSES:
                raise ValueError(f"Unknown test category {category!r} in line {line!r}")

            bounds = [int(x) for x in parts[2:]]
            if len(bounds) % 2:
                bounds = bounds[:-1]  # tolerate a stray trailing token
            intervals = tuple(
                (s, e) for s, e in zip(bounds[0::2], bounds[1::2]) if s != -1 and e != -1
            )
            if category == NORMAL_CLASS:
                intervals = ()
            out.append(TestVideo(name=name, category=category, intervals=intervals))

    if strict:
        n_norm = sum(v.is_normal for v in out)
        if (len(out), n_norm) != (N_TEST, N_TEST_NORMAL):
            raise ValueError(
                f"Test split does not match the official protocol: got {len(out)} videos "
                f"({n_norm} normal), expected {N_TEST} ({N_TEST_NORMAL} normal). "
                f"Frame-level AUC is only comparable to published numbers on the official split."
            )
    return out


def test_split_by_stem(meta_dir: str = DEFAULT_META_DIR) -> Dict[str, TestVideo]:
    """Test videos keyed by filename stem (no extension), for joining against score files."""
    return {v.stem: v for v in load_test_split(meta_dir)}


def dev_subset(
    train: Sequence[TrainVideo],
    test: Sequence[TestVideo],
    n_train_per_class: int = 4,
    n_train_normal: int = 50,
    n_test_anom: int = 20,
    n_test_normal: int = 20,
    seed: int = 0,
) -> Tuple[List[TrainVideo], List[TestVideo]]:
    """
    Small deterministic subset for pipeline development ONLY.

    Results computed on this subset are NOT comparable to published numbers and
    must never appear in a paper. Use it to debug the pipeline, then run the full
    official split for every reported figure.
    """
    rng = np.random.default_rng(seed)

    def pick(items, k):
        items = sorted(items, key=lambda v: v.name)
        if k >= len(items):
            return items
        idx = rng.choice(len(items), size=k, replace=False)
        return [items[i] for i in sorted(idx)]

    tr: List[TrainVideo] = []
    for c in ANOMALY_CLASSES:
        tr += pick([v for v in train if v.category == c], n_train_per_class)
    tr += pick([v for v in train if not v.is_anomalous], n_train_normal)

    te = pick([v for v in test if not v.is_normal], n_test_anom)
    te += pick([v for v in test if v.is_normal], n_test_normal)
    return tr, te


def summarise(meta_dir: str = DEFAULT_META_DIR) -> str:
    """Human-readable split summary, for logging into experiment trackers."""
    train, test = load_train_split(meta_dir), load_test_split(meta_dir)
    lines = [
        "UCF-Crime official split",
        f"  train : {len(train):4d} videos  ({sum(v.is_anomalous for v in train)} anomalous, "
        f"{sum(not v.is_anomalous for v in train)} normal)  [video-level labels only]",
        f"  test  : {len(test):4d} videos  ({sum(not v.is_normal for v in test)} anomalous, "
        f"{sum(v.is_normal for v in test)} normal)  [frame-level labels]",
        f"  test videos with 2 anomaly intervals: {sum(len(v.intervals) > 1 for v in test)}",
        "  per-category test counts:",
    ]
    for c in ANOMALY_CLASSES:
        lines.append(f"    {c:16s} {sum(v.category == c for v in test):3d}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", default=DEFAULT_META_DIR)
    print(summarise(ap.parse_args().meta_dir))
