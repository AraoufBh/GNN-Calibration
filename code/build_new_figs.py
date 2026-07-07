import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = json.load(open("data/results_shift_sweep.json"))
C = json.load(open("data/results_structural.json"))
order = sorted(B.keys(), key=lambda k: B[k]["homophily"])
cols = plt.cm.viridis(np.linspace(0, 0.9, len(order)))

# ---- Fig 11: covariate-shift sweep (Prop 4) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
for k, c in zip(order, cols):
    lv = B[k]["levels"]
    gap = [B[k]["curve"][str(l)]["conf"] - B[k]["curve"][str(l)]["acc"] for l in lv]
    a1.plot(lv, gap, "o-", color=c, label=f"{k} (h={B[k]['homophily']:.2f})")
a1.axhline(0, color="gray", lw=.8, ls=":")
a1.set_xlabel("covariate-shift strength $\\gamma$ (rel. noise)")
a1.set_ylabel("confidence $-$ accuracy")
a1.set_title("(a) Covariate shift $\\Rightarrow$ growing over-confidence (Prop. 4)")
a1.legend(fontsize=7); a1.grid(alpha=.3)
for k, c in zip(order, cols):
    lv = B[k]["levels"]
    unc = [B[k]["curve"][str(l)]["unc"] for l in lv]
    orc = [B[k]["curve"][str(l)]["orc"] for l in lv]
    a2.plot(lv, unc, "-", color=c, lw=2)
    a2.plot(lv, orc, "--", color=c, lw=1.6)
a2.set_xlabel("covariate-shift strength $\\gamma$")
a2.set_ylabel("ECE")
a2.set_title("(b) ECE: uncalibrated (solid) vs. oracle-$T$ (dashed)")
a2.plot([], [], "k-", lw=2, label="uncalibrated")
a2.plot([], [], "k--", lw=1.6, label="oracle $T$")
a2.legend(fontsize=8); a2.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figures/fig11_real_covariate_sweep.png", dpi=140)
print("Saved fig11_real_covariate_sweep.png")

# ---- Fig 12: structural (rewiring) shift ----
order2 = sorted(C.keys(), key=lambda k: B[k]["homophily"])
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for k, c in zip(order2, cols):
    fr = C[k]["fracs"]
    unc = [C[k]["curve"][str(f)]["unc"] for f in fr]
    orc = [C[k]["curve"][str(f)]["orc"] for f in fr]
    ax.plot(fr, unc, "-", color=c, lw=2, label=f"{k}")
    ax.plot(fr, orc, "--", color=c, lw=1.6)
ax.set_xlabel("fraction of edges rewired")
ax.set_ylabel("ECE")
ax.set_title("Structural shift: uncalibrated (solid) vs. oracle-$T$ (dashed)")
ax.plot([], [], "k-", lw=2, label="uncalibrated")
ax.plot([], [], "k--", lw=1.6, label="oracle $T$")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figures/fig12_real_structural.png", dpi=140)
print("Saved fig12_real_structural.png")
