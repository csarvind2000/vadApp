"""One panel per anomaly category: frame at the annotated event, with prediction."""
import json, sys, glob
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,".")
from src.ucfcrime.splits import test_split_by_stem, ANOMALY_CLASSES

plt.rcParams.update({"font.family":"serif","figure.dpi":300,
    "savefig.bbox":"tight","savefig.pad_inches":0.04})
VID="/media/ranjhaa-local/volume23/LLM/UCF-Crime/Videos"
pred=json.load(open("runs/s_milonly_seed0/test_scores.json"))
meta=json.load(open("data/ucfcrime_meta/video_meta.json"))
tv=test_split_by_stem()
GT_C,PR_C,MISS="#2F6DB5","#C44E52","#8a8a8a"

def fs(stem):
    n=int(meta[stem]["n_frames_nominal"])
    s=np.repeat(np.array(pred[stem]),16)
    return np.concatenate([s,np.full(max(0,n-len(s)),s[-1])])[:n],n

def best_for(cat):
    """Pick the video whose annotated interval the model scores highest -- the
    representative *detected* example for that category."""
    best=None
    for stem,v in tv.items():
        if v.category!=cat: continue
        s,n=fs(stem); g=v.frame_labels(n)
        if g.sum()==0: continue
        inside=s[g>0]
        val=float(inside.mean())
        if best is None or val>best[0]: best=(val,stem)
    return best

rows=[]
for c in ANOMALY_CLASSES:
    b=best_for(c)
    if b: rows.append((c,b[1],b[0]))

NC=5; NR=int(np.ceil(len(rows)/NC))
fig,axes=plt.subplots(NR,NC,figsize=(7.2,1.62*NR))
axes=np.atleast_2d(axes)
for ax in axes.ravel(): ax.axis("off")

for i,(cat,stem,mscore) in enumerate(rows):
    ax=axes[i//NC, i%NC]
    s,n=fs(stem); g=tv[stem].frame_labels(n); fps=meta[stem]["fps"] or 30
    idx=np.where(g>0)[0]
    pick=int(idx[np.argmax(s[idx])])            # best-scored annotated frame
    h=glob.glob(f"{VID}/**/{stem}.mp4",recursive=True)
    if not h: continue
    cap=cv2.VideoCapture(h[0]); cap.set(cv2.CAP_PROP_POS_FRAMES,pick)
    ok,fr=cap.read(); cap.release()
    if not ok: continue
    fr=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
    hh,ww=fr.shape[:2]; fr=cv2.resize(fr,(220,max(1,int(hh*220/ww))))
    ax.imshow(fr); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    det = s[pick]>0.5
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(2.4)
        sp.set_color(PR_C if det else MISS)
    ax.set_title(cat,fontsize=7.4,pad=2.5)
    ax.text(0.03,0.965,f"score {s[pick]:.2f}",transform=ax.transAxes,ha="left",va="top",
            fontsize=5.8,color="white",
            bbox=dict(boxstyle="square,pad=0.16",fc=PR_C if det else "#4a4a4a",ec="none",alpha=0.95))
    ax.text(0.03,0.035,"detected" if det else "missed",transform=ax.transAxes,
            ha="left",va="bottom",fontsize=5.6,color="white",
            bbox=dict(boxstyle="square,pad=0.16",fc=GT_C,ec="none",alpha=0.9))
    ax.set_xlabel(f"{stem.replace('_x264','')}  t={pick/fps:.1f}s",fontsize=5.6,labelpad=1.6)
fig.tight_layout(h_pad=1.1,w_pad=0.5)
fig.savefig("manuscript/figures/fig_categories.png",dpi=300); plt.close(fig)
print(f"fig_categories.png: {len(rows)} categories")
det=sum(1 for c,s_,m in rows if m>0.5)
for c,stem,m in rows: print(f"  {c:15s} {stem:24s} mean-score-in-GT={m:.3f}")
