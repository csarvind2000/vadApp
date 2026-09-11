"""Mechanism panels: dilated receptive fields, entmax vs softmax, MIL top-k."""
import numpy as np, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
sys.path.insert(0,".")
import torch
from src.ucfcrime.ata_hs import entmax15, sparsemax

plt.rcParams.update({"font.family":"serif","font.size":8,"axes.linewidth":0.8,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":300,
    "savefig.bbox":"tight","savefig.pad_inches":0.03})
BL,RD,OR,GR,GY="#4C72B0","#C44E52","#DD8452","#55A868","#9a9a9a"

fig=plt.figure(figsize=(7.2,5.9))
gs=fig.add_gridspec(3,2,hspace=0.95,wspace=0.26,height_ratios=[1.15,1,1])

# ---------- (a) dilated temporal convolution receptive fields -------------
ax=fig.add_subplot(gs[0,:])
T=21; ctr=10
rows=[(1,3),(2,5),(4,9),(8,17)]
for r,(d,rf) in enumerate(rows):
    yy=len(rows)-1-r
    for t in range(T):
        ax.add_patch(Rectangle((t-0.42,yy-0.30),0.84,0.60,
                     facecolor="#eef1f5",edgecolor="#c8ced6",linewidth=0.4))
    taps=[ctr-d,ctr,ctr+d]                       # kernel size 3 at this dilation
    for t in taps:
        ax.add_patch(Rectangle((t-0.42,yy-0.30),0.84,0.60,
                     facecolor=BL,edgecolor="none",alpha=0.85))
    lo,hi=ctr-(rf//2),ctr+(rf//2)                # cumulative receptive field
    ax.add_patch(Rectangle((lo-0.46,yy-0.38),(hi-lo)+0.92,0.76,
                 facecolor="none",edgecolor=RD,linewidth=1.0,linestyle="--"))
    ax.text(-1.4,yy,f"dilation {d}",ha="right",va="center",fontsize=7)
    ax.text(T+0.4,yy,f"RF {rf} snippets $\\approx$ {rf*16/30:.1f} s",
            ha="left",va="center",fontsize=6.4,color=RD)
ax.set_xlim(-6,T+9); ax.set_ylim(-0.8,len(rows)-0.2)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("(a) Multi-scale dilated temporal convolution: four parallel branches, kernel size 3.\n"
             "Blue = sampled taps, dashed = cumulative receptive field. Branch outputs are\n"
             "concatenated, so every snippet sees 1.6 to 9.1 s of surrounding context.",
             fontsize=7.6,loc="left")
ax.annotate("",xy=(ctr,-0.55),xytext=(ctr,-0.15),
            arrowprops=dict(arrowstyle="-",lw=0.7,color=GY))
ax.text(ctr,-0.72,"snippet $t$",ha="center",fontsize=6.4,color=GY)

# ---------- (b) entmax vs softmax ----------------------------------------
ax=fig.add_subplot(gs[1,0])
z=torch.tensor([[3.0,2.6,2.1,1.4,0.9,0.5,0.1,-0.4,-1.0,-1.8]])
sm=torch.softmax(z,1)[0].numpy()
em=entmax15(z,1)[0].numpy()
sp=sparsemax(z,1)[0].numpy()
x=np.arange(z.shape[1]); w=0.27
ax.bar(x-w,sm,w,label="softmax",color=GY)
ax.bar(x,  em,w,label=r"entmax$_{1.5}$ (used)",color=RD)
ax.bar(x+w,sp,w,label="sparsemax",color=BL)
ax.set_xlabel("snippet index (sorted by logit)"); ax.set_ylabel("attention weight")
ax.legend(fontsize=6.2,frameon=False)
ax.set_title(f"(b) Attention sparsity over 10 snippets.\nExact zeros: softmax {int((sm==0).sum())}, "
             f"entmax {int((em==0).sum())}, sparsemax {int((sp==0).sum())}.",
             fontsize=7.4,loc="left")

# ---------- (c) entmax gradient exists at the boundary --------------------
ax=fig.add_subplot(gs[1,1])
gs_=np.linspace(-3,3,220)
sm_out,em_out=[],[]
for g in gs_:
    zz=torch.tensor([[g,0.0,-0.5]])
    sm_out.append(float(torch.softmax(zz,1)[0,0]))
    em_out.append(float(entmax15(zz,1)[0,0]))
ax.plot(gs_,sm_out,color=GY,lw=1.2,label="softmax")
ax.plot(gs_,em_out,color=RD,lw=1.4,label=r"entmax$_{1.5}$")
ax.axhline(0,color="#ccc",lw=0.6)
ax.set_xlabel("logit of snippet 1"); ax.set_ylabel("assigned weight")
ax.legend(fontsize=6.4,frameon=False)
ax.set_title("(c) entmax reaches exactly zero and stays\ndifferentiable; hard top-$K$ is piecewise\n"
             "constant and passes no gradient.",fontsize=7.4,loc="left")

# ---------- (d) MIL head and top-k bag score ------------------------------
ax=fig.add_subplot(gs[2,:])
rng=np.random.default_rng(3)
Tn=26
anom=np.clip(rng.beta(1.4,7,Tn),0,1); anom[11:14]=[0.93,0.97,0.88]
norm=np.clip(rng.beta(1.3,9,Tn),0,1)
xx=np.arange(Tn)
ax.bar(xx-0.2,anom,0.38,color=RD,label="anomalous bag")
ax.bar(xx+0.2,norm,0.38,color=BL,label="normal bag")
k=3
ta=np.argsort(-anom)[:k]; tn=np.argsort(-norm)[:k]
for i in ta: ax.scatter(i-0.2,anom[i]+0.055,marker="v",s=17,color=RD,zorder=4)
for i in tn: ax.scatter(i+0.2,norm[i]+0.055,marker="v",s=17,color=BL,zorder=4)
sa,sn=anom[ta].mean(),norm[tn].mean()
ax.axhline(sa,color=RD,ls="--",lw=0.8); ax.axhline(sn,color=BL,ls="--",lw=0.8)
ax.annotate("",xy=(Tn-0.5,sa),xytext=(Tn-0.5,sn),
            arrowprops=dict(arrowstyle="<->",lw=0.9,color="black"))
ax.text(Tn-0.2,(sa+sn)/2,f"margin\n$s_+-s_-$\n$={sa-sn:.2f}$",fontsize=6.2,va="center")
ax.set_xlim(-1,Tn+3.2); ax.set_ylim(0,1.16)
ax.set_xlabel("snippet index within bag"); ax.set_ylabel("MIL head score $s_t$")
ax.legend(fontsize=6.4,frameon=False,loc="upper left")
ax.set_title("(d) MIL head: per-snippet score from a 512-128-1 MLP with sigmoid. The bag score is the "
             "mean of the\ntop-$k$ snippets ($k=3$, markers). The hinge loss drives the anomalous bag "
             "score above the normal one.",fontsize=7.4,loc="left")

fig.savefig("manuscript/figures/fig_mechanisms.png",dpi=300); plt.close(fig)
print("fig_mechanisms.png written")
