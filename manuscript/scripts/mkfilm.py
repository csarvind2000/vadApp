"""Filmstrip figures: real frames sampled at informative timepoints,
annotated with ground truth and prediction."""
import json, os, sys, glob
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from src.ucfcrime.splits import test_split_by_stem

plt.rcParams.update({"font.family":"serif","font.size":8,"figure.dpi":300,
    "savefig.bbox":"tight","savefig.pad_inches":0.04,
    "axes.spines.top":False,"axes.spines.right":False})
VID="/media/ranjhaa-local/volume23/LLM/UCF-Crime/Videos"
OUT="manuscript/figures"
pred=json.load(open("runs/s_milonly_seed0/test_scores.json"))
meta=json.load(open("data/ucfcrime_meta/video_meta.json"))
tv=test_split_by_stem()
GT_C, PR_C, NEU = "#2F6DB5", "#C44E52", "#9a9a9a"

def frame_scores(stem):
    n=int(meta[stem]["n_frames_nominal"])
    s=np.repeat(np.array(pred[stem]),16)
    return np.concatenate([s,np.full(max(0,n-len(s)),s[-1])])[:n], n

def find(stem):
    h=glob.glob(f"{VID}/**/{stem}.mp4", recursive=True); return h[0] if h else None

def pick_times(s, gt, n, k=6, thr=0.5):
    """Sample informative frames: inside GT intervals, at score peaks, plus context.
    Avoid the first/last 2% of the video, which are frequently blank."""
    lo, hi = int(0.02*n), int(0.98*n)
    cand=set()
    d=np.diff(np.concatenate([[0],(gt>0).astype(int),[0]]))
    for a,b in zip(np.where(d==1)[0], np.where(d==-1)[0]):     # inside each GT interval
        cand.update(int(x) for x in np.linspace(a, max(a,b-1), 3))
    if (s>thr).any():                                          # top prediction peaks
        cand.update(int(x) for x in np.argsort(-s)[:200:40])
    cand.update(int(x) for x in np.linspace(lo, hi, 4))         # context
    out=sorted(x for x in cand if lo<=x<=hi)
    if len(out)>k:                                             # spread evenly
        out=[out[i] for i in np.linspace(0,len(out)-1,k).astype(int)]
    while len(out)<k:
        out.append(min(hi, out[-1]+max(1,n//k)))
    return sorted(set(out))[:k]

def filmstrip(stem, axes, thr=0.5):
    s,n=frame_scores(stem); fps=meta[stem]["fps"] or 30
    gt=tv[stem].frame_labels(n)
    idxs=pick_times(s,gt,n,k=len(axes),thr=thr)
    cap=cv2.VideoCapture(find(stem))
    for ax,fi in zip(axes,idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(fi)); ok,fr=cap.read()
        if not ok: ax.axis("off"); continue
        fr=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
        h,w=fr.shape[:2]; fr=cv2.resize(fr,(200,max(1,int(h*200/w))))
        ax.imshow(fr); ax.set_xticks([]); ax.set_yticks([])
        g=bool(gt[fi]); p=bool(s[fi]>thr)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(2.6)
            sp.set_color(GT_C if g else NEU)
        ax.text(0.02,0.98,"GT: anom" if g else "GT: norm",transform=ax.transAxes,
                ha="left",va="top",fontsize=5.4,color="white",
                bbox=dict(boxstyle="square,pad=0.16",fc=GT_C if g else "#4a4a4a",ec="none",alpha=0.95))
        ax.text(0.02,0.02,f"PR: {s[fi]:.2f}",transform=ax.transAxes,
                ha="left",va="bottom",fontsize=5.4,color="white",
                bbox=dict(boxstyle="square,pad=0.16",fc=PR_C if p else "#4a4a4a",ec="none",alpha=0.95))
        ax.set_xlabel(f"{fi/fps:.1f}s",fontsize=6.0,labelpad=1.5)
    cap.release(); return s,gt,fps,n,idxs

def curve(ax,s,gt,fps,n,marks,thr=0.5,legend=False):
    t=np.arange(n)/fps
    d=np.diff(np.concatenate([[0],(gt>0).astype(int),[0]]))
    for i,(a,b) in enumerate(zip(np.where(d==1)[0],np.where(d==-1)[0])):
        ax.axvspan(a/fps,b/fps,color=GT_C,alpha=0.25,lw=0,
                   label="ground-truth anomaly" if i==0 else None)
    ax.plot(t,s,color=PR_C,lw=1.0,label="predicted score")
    ax.axhline(thr,ls="--",lw=0.7,color="#DD8452")
    for m in marks: ax.axvline(m/fps,color="#444",lw=0.5,alpha=0.5,ls=":")
    ax.set_ylim(-0.05,1.08); ax.set_xlim(0,t[-1])
    ax.set_ylabel("score",fontsize=7); ax.set_xlabel("time (s)",fontsize=7)
    ax.tick_params(labelsize=6.3)
    if legend: ax.legend(fontsize=6.0,frameon=False,ncol=2,loc="upper left")

picks=[("Shooting015_x264","(a) Correct detection"),
       ("Abuse028_x264","(b) Missed detection"),
       ("Normal_Videos_478_x264","(c) False alarm, normal video")]
NC=6
fig=plt.figure(figsize=(7.2,2.85*len(picks)))
gs=fig.add_gridspec(len(picks)*2,NC,height_ratios=sum([[2.5,1.0] for _ in picks],[]),
                    hspace=0.62,wspace=0.07)
for r,(stem,title) in enumerate(picks):
    axr=[fig.add_subplot(gs[r*2,c]) for c in range(NC)]
    s,gt,fps,n,idxs=filmstrip(stem,axr)
    axr[0].text(-0.16,0.5,title,transform=axr[0].transAxes,rotation=90,
                va="center",ha="right",fontsize=7.6)
    axc=fig.add_subplot(gs[r*2+1,:])
    curve(axc,s,gt,fps,n,idxs,legend=(r==0))
    axc.set_title(f"{stem}  ({tv[stem].category})",fontsize=7.4,loc="left",pad=2)
fig.savefig(f"{OUT}/fig_visual_results.png",dpi=300); plt.close(fig)
print("written")
for stem,_ in picks:
    s,n=frame_scores(stem); gt=tv[stem].frame_labels(n)
    print(f"  {stem}: GT positive frames={int(gt.sum())}/{n}, pred>0.5={(s>0.5).sum()}")
