"""Generate result figures from the actual experiment outputs."""
import json, glob, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
OUT = "manuscript/figures"
os.makedirs(OUT, exist_ok=True)
C = {"base": "#4C72B0", "ours": "#C44E52", "grey": "#7f7f7f", "acc": "#DD8452"}
EK = dict(elinewidth=0.9)

runs = collections.defaultdict(list)
for p in sorted(glob.glob("runs/s_*_seed*/report.json")):
    tag = os.path.basename(os.path.dirname(p)).replace("s_", "").rsplit("_seed", 1)[0]
    runs[tag].append(json.load(open(p)))
ORDER = ["sultani", "full", "softmax", "milonly", "nomulti", "nobudget"]
LABEL = {"sultani": "MIL\nbaseline", "full": "ATA-HS\n(full)", "softmax": "$-$entmax",
         "milonly": "$-$hybrid\nheads", "nomulti": "$-$multi-\nscale",
         "nobudget": "$-$adaptive\nbudget"}
def arr(t, k): return np.array([r[k] for r in runs[t]])

# ---- Fig: ablation (AUC + FAR) -------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
fig.subplots_adjust(wspace=0.32)
x = np.arange(len(ORDER))
cols = [C["base"]] + [C["ours"]] * (len(ORDER) - 1)
m = [arr(t, "auc_micro").mean() for t in ORDER]
s = [arr(t, "auc_micro").std(ddof=1) for t in ORDER]
ax[0].bar(x, m, yerr=s, capsize=3, color=cols, width=0.62, error_kw=EK)
ax[0].axhline(m[0], ls="--", lw=0.8, color=C["grey"])
ax[0].set_xticks(x); ax[0].set_xticklabels([LABEL[t] for t in ORDER], fontsize=6.2)
ax[0].set_ylabel("Frame-level AUC"); ax[0].set_ylim(0.65, 0.82)
ax[0].set_title("(a) Detection accuracy", fontsize=8.5)

mf = [arr(t, "far@0.5").mean() for t in ORDER]
sf = [arr(t, "far@0.5").std(ddof=1) for t in ORDER]
ax[1].bar(x, mf, yerr=sf, capsize=3, color=cols, width=0.62, error_kw=EK)
ax[1].axhline(mf[0], ls="--", lw=0.8, color=C["grey"])
ax[1].set_xticks(x); ax[1].set_xticklabels([LABEL[t] for t in ORDER], fontsize=6.2)
ax[1].set_ylabel("False alarm rate")
ax[1].set_title("(b) False alarms on normal video", fontsize=8.5)
fig.tight_layout(w_pad=2.2)
fig.savefig(f"{OUT}/fig_ablation.png", dpi=300); plt.close(fig)

# ---- Fig: single-seed vs multi-seed (why one run misleads) ----------------
sw = {}
for line in open("/tmp/claude-1001/sweep2.log"):
    if "->" in line:
        k, v = line.split("->"); sw[k.strip()] = float(v.split()[0])
pairs = [("full", sw.get("k1")), ("softmax", sw.get("k3_softmax")),
         ("milonly", sw.get("k3_milonly")), ("nomulti", sw.get("k3_nomulti")),
         ("nobudget", sw.get("k3_nobudget"))]
pairs = [(a, b) for a, b in pairs if b is not None]
fig, ax = plt.subplots(figsize=(3.4, 2.4))
xs = np.arange(len(pairs))
ax.bar(xs - 0.19, [b for _, b in pairs], 0.38, label="single seed", color=C["grey"])
ax.bar(xs + 0.19, [arr(a, "auc_micro").mean() for a, _ in pairs], 0.38,
       yerr=[arr(a, "auc_micro").std(ddof=1) for a, _ in pairs], capsize=2.5,
       label="mean of 3 seeds", color=C["ours"], error_kw=EK)
ax.set_xticks(xs); ax.set_xticklabels([LABEL[a].replace("\n", " ") for a, _ in pairs],
                                      fontsize=6.2, rotation=20, ha="right")
ax.set_ylabel("Frame-level AUC"); ax.set_ylim(0.65, 0.82)
ax.legend(fontsize=6.5, frameon=False)
fig.savefig(f"{OUT}/fig_seed_effect.png", dpi=300); plt.close(fig)

# ---- Fig: bootstrap CI vs published leaderboard --------------------------
rep = json.load(open("runs/ucf_sultani_i3d_s0/report.json"))
ci = rep.get("auc_micro_ci", {"lo": 0.6961, "hi": 0.8489})
lit = [("Sultani'18", 75.41), ("GCN'19", 82.12), ("RTFM'21", 84.30),
       ("MSL'22", 85.30), ("S3R'22", 85.99), ("UR-DMU'23", 86.97),
       ("MGFN'23", 86.98), ("CLIP-TSA'23", 87.58), ("VadCLIP'24", 88.02)]
fig, ax = plt.subplots(figsize=(6.6, 2.3))
xs = np.arange(len(lit))
ax.plot(xs, [v for _, v in lit], "o-", color=C["base"], ms=4, lw=1.2)
half = 100 * (ci["hi"] - ci["lo"]) / 2
ax.errorbar([0], [75.41], yerr=[half], color=C["ours"], capsize=4, lw=1.4,
            label=f"95% CI measured here ($\\pm${half:.1f} AUC)")
ax.axhspan(75.41 - half, 75.41 + half, color=C["ours"], alpha=0.12)
ax.set_xticks(xs); ax.set_xticklabels([n for n, _ in lit], fontsize=6.3, rotation=25, ha="right")
ax.set_ylabel("Reported frame-level AUC (%)")
ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.savefig(f"{OUT}/fig_ci_vs_literature.png", dpi=300); plt.close(fig)

# ---- Fig: validation does not predict test ------------------------------
v, t = [], []
for h in sorted(glob.glob("runs/s_*_seed*/history.json")):
    d = os.path.dirname(h); r = os.path.join(d, "report.json")
    if not os.path.isfile(r): continue
    v.append(json.load(open(h))["best_val_auc"]); t.append(json.load(open(r))["auc_micro"])
v, t = np.array(v), np.array(t)
from scipy.stats import pearsonr
pr, pp = pearsonr(v, t)
fig, ax = plt.subplots(figsize=(3.2, 2.4))
ax.scatter(v, t, s=22, color=C["ours"], edgecolor="white", linewidth=0.5, zorder=3)
z = np.polyfit(v, t, 1)
xx = np.linspace(v.min(), v.max(), 20)
ax.plot(xx, np.polyval(z, xx), "--", lw=1, color=C["grey"])
ax.set_xlabel("Validation video-level AUC"); ax.set_ylabel("Test frame-level AUC")
ax.set_title(f"$r={pr:+.2f}$ ($p={pp:.2f}$, $n={len(v)}$)", fontsize=8)
fig.savefig(f"{OUT}/fig_val_vs_test.png", dpi=300); plt.close(fig)

# ---- Fig: per-category AUC ----------------------------------------------
cat = collections.defaultdict(list)
for r in runs["sultani"] + runs["milonly"]:
    for k, val in r["per_category_auc"].items():
        cat[k].append(val)
ks = sorted(cat, key=lambda k: -np.mean(cat[k]))
fig, ax = plt.subplots(figsize=(6.6, 2.3))
ax.bar(np.arange(len(ks)), [np.mean(cat[k]) for k in ks],
       yerr=[np.std(cat[k], ddof=1) for k in ks], capsize=2.5,
       color=C["base"], width=0.65, error_kw=EK)
ax.axhline(0.5, ls=":", lw=0.8, color=C["grey"])
ax.set_xticks(np.arange(len(ks))); ax.set_xticklabels(ks, rotation=35, ha="right", fontsize=6.5)
ax.set_ylabel("AUC"); ax.set_ylim(0.4, 1.0)
fig.savefig(f"{OUT}/fig_per_category.png", dpi=300); plt.close(fig)
print("figures written:", sorted(os.listdir(OUT)))
