"""
Experiment A: does the closed form predict the OPTIMAL temperature, not just its sign?
For a grid of (h_s, h_t, rho) we compare the predicted T* = 1/kappa(h_t; h_s, rho) with the
oracle temperature T_oracle measured on the (labeled) target. A tight y=x scatter shows the
theory is quantitatively predictive.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from theory import (sample_binary_csbm, mean_aggregate, opt_temperature,
                    kappa_pred, edge_homophily)


def run(N=1500, deg=16, seeds=4):
    ps = deg * 2.0 / N
    pred, meas, rhos = [], [], []
    for rho in (0.5, 1.0, 2.0):
        sigma = 1.0 / np.sqrt(rho)
        hs = 0.75
        for ht in np.linspace(0.57, 0.88, 7):
            Tp, To = [], []
            for s in range(seeds):
                rng = np.random.default_rng(7 * s + int(1000 * ht) + int(10 * rho))
                A_s, x_s, y_s = sample_binary_csbm(N, ps*hs, ps*(1-hs), 1.0, sigma, rng)
                z_s, _ = mean_aggregate(A_s, x_s); y01s = (y_s > 0).astype(int)
                clf = LogisticRegression(C=1e6, solver="lbfgs").fit(z_s.reshape(-1, 1), y01s)
                Ls = clf.decision_function(z_s.reshape(-1, 1))
                Tsrc = opt_temperature(Ls, y01s); hsm = edge_homophily(A_s, y_s)
                A_t, x_t, y_t = sample_binary_csbm(N, ps*ht, ps*(1-ht), 1.0, sigma, rng)
                z_t, _ = mean_aggregate(A_t, x_t); y01t = (y_t > 0).astype(int)
                Lt = clf.decision_function(z_t.reshape(-1, 1)) / Tsrc
                htm = edge_homophily(A_t, y_t)
                k = kappa_pred(htm, hsm, rho)
                To.append(opt_temperature(Lt, y01t)); Tp.append(1.0 / k)
            pred.append(np.mean(Tp)); meas.append(np.mean(To)); rhos.append(rho)
        print(f"  rho={rho} done", flush=True)
    return np.array(pred), np.array(meas), np.array(rhos)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pred, meas, rhos = run()
    # restrict to the well-conditioned range (oracle T finite, not saturated)
    ok = (meas > 0.3) & (meas < 6) & (pred > 0.3) & (pred < 6)
    p, m = pred[ok], meas[ok]
    mae = np.mean(np.abs(p - m)); r = np.corrcoef(p, m)[0, 1]
    print(f"points={ok.sum()}  MAE(T)={mae:.3f}  Pearson r={r:.3f}")
    fig, ax = plt.subplots(figsize=(5.2, 5))
    cols = {0.5: "#1f77b4", 1.0: "#2ca02c", 2.0: "#d62728"}
    for rr in (0.5, 1.0, 2.0):
        sel = ok & (rhos == rr)
        ax.scatter(meas[sel], pred[sel], s=42, color=cols[rr], alpha=.8,
                   edgecolor="k", linewidth=.4, label=f"$\\rho={rr}$")
    lim = [0.5, 3.0]
    ax.plot(lim, lim, "k--", lw=1, label="$y=x$")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("oracle temperature $T_{\\mathrm{oracle}}$")
    ax.set_ylabel("predicted $T^\\star=1/\\kappa$")
    ax.set_title(f"Closed form predicts the optimal temperature\nMAE={mae:.3f}, $r$={r:.3f}")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("figures/fig10_temp_scatter.png", dpi=140)
    print("Saved fig10_temp_scatter.png")
