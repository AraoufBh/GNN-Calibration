import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("data/results_real.json"))
order = sorted(R.keys(), key=lambda k: R[k]["homophily"])
methods = ["Uncal", "Source-TS", "STAC", "Oracle"]
cols = ["#888", "#d62728", "#1f77b4", "#2ca02c"]

fig, ax = plt.subplots(figsize=(11, 4.6))
x = np.arange(len(order)); w = 0.2
for j, (m, c) in enumerate(zip(methods, cols)):
    mu = [R[k][m][0] for k in order]; sd = [R[k][m][1] for k in order]
    ax.bar(x + (j - 1.5) * w, mu, w, yerr=sd, capsize=2, label=m, color=c)
ax.set_xticks(x)
ax.set_xticklabels([f"{k}\n(h={R[k]['homophily']:.2f}, K={R[k]['K']})" for k in order],
                   fontsize=8)
ax.set_ylabel("ECE (\u2193)")
ax.set_title("Real graphs under covariate shift: one Oracle temperature recovers "
             "calibration everywhere;\nlabel-free STAC is unreliable (accuracy-estimation bottleneck)")
ax.legend(); ax.grid(alpha=.3, axis="y")
fig.tight_layout()
fig.savefig("figures/fig9_real_multidataset.png", dpi=140)
print("Saved fig9_real_multidataset.png")
