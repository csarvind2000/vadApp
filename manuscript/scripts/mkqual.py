import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from src.ucfcrime.splits import test_split_by_stem
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.linewidth":0.8,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":300,
    "savefig.bbox":"tight","savefig.pad_inches":0.02})
C={"score":"#C44E52","gt":"#4C72B0","thr":"#DD8452"}
pred=json.load(open("runs/s_milonly_seed0/test_scores.json"))
meta=json.load(open("data/ucfcrime_meta/video_meta.json"))
tv=test_split_by_stem()

# two successes and one failure, chosen by localisation quality
def iou(stem):
    v=tv[stem]; n=int(meta[stem]["n_frames_nominal"]); g=v.frame_labels(n)
    s=np.repeat(np.array(pred[stem]),16)
    s=np.concatenate([s,np.full(max(0,n-len(s)),s[-1])])[:n]
    p=s>0.5
    inter=(p&(g>0)).sum(); union=(p|(g>0)).sum()
    return inter/max(union,1)
cands=[k for k in pred if not tv[k].is_normal]
scored=sorted(((iou(k),k) for k in cands), reverse=True)
picks=[scored[0][1], scored[1][1], scored[-1][1]]
normal=[k for k in pred if tv[k].is_normal and np.array(pred[k]).max()>0.9]
if normal: picks.append(normal[0])

fig,axes=plt.subplots(len(picks),1,figsize=(6.6,1.35*len(picks)),sharex=False)
for ax,stem in zip(np.atleast_1d(axes),picks):
    v=tv[stem]; n=int(meta[stem]["n_frames_nominal"]); fps=meta[stem]["fps"] or 30
    g=v.frame_labels(n)
    s=np.repeat(np.array(pred[stem]),16)
    s=np.concatenate([s,np.full(max(0,n-len(s)),s[-1])])[:n]
    t=np.arange(n)/fps
    # ground-truth intervals as shaded bands
    d=np.diff(np.concatenate([[0],(g>0).astype(int),[0]]))
    for a,b in zip(np.where(d==1)[0], np.where(d==-1)[0]):
        ax.axvspan(a/fps, b/fps, color=C["gt"], alpha=0.22, lw=0,
                   label="ground truth" if a==np.where(d==1)[0][0] else None)
    ax.plot(t,s,color=C["score"],lw=0.9,label="predicted score")
    ax.axhline(0.5,ls="--",lw=0.7,color=C["thr"])
    ax.set_ylim(-0.05,1.05); ax.set_xlim(0,t[-1])
    ax.set_ylabel("score",fontsize=7)
    kind = "normal video (false alarm)" if v.is_normal else f"{v.category}"
    ax.set_title(f"{stem}  —  {kind}   IoU={iou(stem):.2f}" if not v.is_normal
                 else f"{stem}  —  {kind}", fontsize=7.5, loc="left")
    ax.tick_params(labelsize=6.5)
np.atleast_1d(axes)[-1].set_xlabel("time (s)",fontsize=7.5)
h,l=np.atleast_1d(axes)[0].get_legend_handles_labels()
np.atleast_1d(axes)[0].legend(h,l,fontsize=6.2,frameon=False,ncol=2,loc="upper right")
fig.tight_layout()
fig.savefig("manuscript/figures/fig_qualitative.png", dpi=300); plt.close(fig)
print("fig_qualitative.pdf ->", picks)
print("IoU stats over 140 anomalous test videos:")
ious=np.array([iou(k) for k in cands])
print(f"  mean={ious.mean():.3f} median={np.median(ious):.3f} >0.5: {(ious>0.5).sum()}/{len(ious)}")
