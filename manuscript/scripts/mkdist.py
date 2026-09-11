"""UCF-Crime composition: category balance, event extent, duration."""
import json, sys, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,".")
from src.ucfcrime.splits import load_train_split, load_test_split, ANOMALY_CLASSES

plt.rcParams.update({"font.family":"serif","font.size":8,"axes.linewidth":0.8,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":300,
    "savefig.bbox":"tight","savefig.pad_inches":0.03})
TR_C,TE_C,AC="#4C72B0","#C44E52","#DD8452"
tr=load_train_split(); te=load_test_split()
meta=json.load(open("data/ucfcrime_meta/video_meta.json"))
ctr=collections.Counter(v.category for v in tr if v.is_anomalous)
cte=collections.Counter(v.category for v in te if not v.is_normal)
cats=sorted(ANOMALY_CLASSES, key=lambda c:-ctr[c])

fig=plt.figure(figsize=(7.2,4.9))
gs=fig.add_gridspec(2,2,hspace=0.62,wspace=0.28,height_ratios=[1,1])

# (a) train vs test counts -- exposes the inverted balance
ax=fig.add_subplot(gs[0,:])
x=np.arange(len(cats)); w=0.38
ax.bar(x-w/2,[ctr[c] for c in cats],w,label=f"train ({sum(ctr.values())} videos)",color=TR_C)
ax.bar(x+w/2,[cte[c] for c in cats],w,label=f"test ({sum(cte.values())} videos)",color=TE_C)
for i,c in enumerate(cats):
    ax.text(i-w/2,ctr[c]+2,str(ctr[c]),ha="center",fontsize=5.6,color=TR_C)
    ax.text(i+w/2,cte[c]+2,str(cte[c]),ha="center",fontsize=5.6,color=TE_C)
ax.set_xticks(x); ax.set_xticklabels(cats,rotation=32,ha="right",fontsize=6.8)
ax.set_ylabel("videos"); ax.legend(fontsize=6.8,frameon=False)
ax.set_title("(a) Anomalous videos per category. Training and test balance differ: "
             "Robbery 145/5, Shooting 27/23.",fontsize=8,loc="left")

# (b) fraction of each test video that is annotated anomalous
ax=fig.add_subplot(gs[1,0])
frac=[]
for v in te:
    if v.is_normal: continue
    n=meta[v.stem]["n_frames_nominal"]
    frac.append(v.frame_labels(n).sum()/n)
frac=np.array(frac)
ax.hist(frac,bins=22,color=AC,edgecolor="white",linewidth=0.4)
ax.axvline(frac.mean(),color=TE_C,ls="--",lw=1,
           label=f"mean {frac.mean():.2f}")
ax.axvline(np.median(frac),color=TR_C,ls=":",lw=1,
           label=f"median {np.median(frac):.2f}")
ax.set_xlabel("anomalous fraction of video"); ax.set_ylabel("test videos")
ax.legend(fontsize=6.3,frameon=False)
ax.set_title("(b) Event extent",fontsize=8,loc="left")

# (c) per-category anomalous fraction -- how much of a video the event occupies
ax=fig.add_subplot(gs[1,1])
byc=collections.defaultdict(list)
for v in te:
    if v.is_normal: continue
    n=meta[v.stem]["n_frames_nominal"]
    byc[v.category].append(v.frame_labels(n).sum()/n)
ks=sorted([c for c in ANOMALY_CLASSES if byc[c]], key=lambda c: np.mean(byc[c]))
y=np.arange(len(ks))
ax.barh(y,[np.mean(byc[c]) for c in ks],
        xerr=[np.std(byc[c]) if len(byc[c])>1 else 0 for c in ks],
        color=AC,height=0.66,error_kw=dict(elinewidth=0.7,ecolor="#8a8a8a"))
ax.axvline(frac.mean(),color=TE_C,ls="--",lw=0.9)
ax.set_yticks(y); ax.set_yticklabels(ks,fontsize=6.2)
ax.set_xlabel("anomalous fraction of video")
ax.set_title("(c) Event extent by category",fontsize=8,loc="left")

fig.savefig("manuscript/figures/fig_dataset.png",dpi=300); plt.close(fig)
print("fig_dataset.png written")
