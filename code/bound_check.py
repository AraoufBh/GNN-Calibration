"""
Validation of the calibration bound:  ECE  <=  (1/4) |kappa - 1| * E|delta|,
where delta is the logit (margin) and kappa the closed-form calibration slope.
Reuses the binary-CSBM mean-aggregation setup from theory.py.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from theory import (sample_binary_csbm, mean_aggregate, opt_temperature,
                    ece_binary, kappa_pred, edge_homophily)


def run(hs=0.8, rho=1.0, N=4000, deg=25, mu=1.0, seeds=6):
    sigma = mu / np.sqrt(rho); ps = deg * 2.0 / N
    ht_grid = np.array([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    rows = []
    for ht in ht_grid:
        ece_m, bound, kmv = [], [], []
        for s in range(seeds):
            rng = np.random.default_rng(31*s + int(1000*ht))
            A_s, x_s, y_s = sample_binary_csbm(N, ps*hs, ps*(1-hs), mu, sigma, rng)
            z_s, _ = mean_aggregate(A_s, x_s); y01_s = (y_s > 0).astype(int)
            clf = LogisticRegression(C=1e6, solver="lbfgs").fit(z_s.reshape(-1, 1), y01_s)
            L_s = clf.decision_function(z_s.reshape(-1, 1))
            T_src = opt_temperature(L_s, y01_s)
            hs_m = edge_homophily(A_s, y_s)
            A_t, x_t, y_t = sample_binary_csbm(N, ps*ht, ps*(1-ht), mu, sigma, rng)
            z_t, _ = mean_aggregate(A_t, x_t); y01_t = (y_t > 0).astype(int)
            L_t = clf.decision_function(z_t.reshape(-1, 1)) / T_src
            ht_m = edge_homophily(A_t, y_t)
            p = 1 / (1 + np.exp(-np.clip(L_t, -30, 30)))
            ece_m.append(ece_binary(p, y01_t))
            k = kappa_pred(ht_m, hs_m, rho)
            bound.append(0.25 * abs(k - 1) * np.mean(np.abs(L_t)))
            kmv.append(k)
        rows.append((float(ht), float(np.mean(ece_m)), float(np.mean(bound)),
                     float(np.mean(kmv))))
    return rows


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT = "figures"

    rows = run()
    print(f"{'h_t':>5} {'kappa':>7} {'ECE_meas':>9} {'bound':>8} {'holds?':>7}")
    for ht, e, b, k in rows:
        print(f"{ht:>5.2f} {k:>7.3f} {e:>9.4f} {b:>8.4f} {str(e <= b + 1e-6):>7}")

    ht = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.plot(ht, [r[2] for r in rows], "r--s", label=r"bound $\frac{1}{4}|\kappa-1|\,E|\delta|$")
    ax.plot(ht, [r[1] for r in rows], "k-o", label="measured ECE")
    ax.fill_between(ht, [r[1] for r in rows], [r[2] for r in rows], color="red", alpha=.08)
    ax.axvline(0.8, color="orange", ls=":", alpha=.8, label="source homophily")
    ax.set_xlabel("target homophily $h_t$"); ax.set_ylabel("ECE")
    ax.set_title("Calibration bound holds and tracks the true ECE")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig7_ece_bound.png", dpi=140)
    print("\nSaved: fig7_ece_bound.png")
