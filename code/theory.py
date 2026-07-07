"""
Numerical validation of the calibration-slope theorem for a linear GNN under
homophily shift, on a binary Contextual SBM.

Theory (mean-aggregation SGC, classes +/-1, x_i = y_i*mu + N(0,sigma^2 I)):
  At logit t, model confidence = sigmoid(t) but true accuracy = sigmoid(kappa*t),
  with the closed-form calibration slope
      kappa(h_t) = (2h_t-1)(1+4 h_s(1-h_s) rho) / [ (2h_s-1)(1+4 h_t(1-h_t) rho) ]
  where h_s,h_t are source/target edge-homophily and rho=||mu||^2/sigma^2 (SNR).
  => over-confident iff kappa<1, under-confident iff kappa>1, calibrated iff kappa=1,
  => optimal recalibration temperature T* = 1/kappa.

We verify kappa_pred against kappa_measured := 1 / T_oracle, where T_oracle is the
NLL-optimal temperature on the (label-bearing) target graph. A source temperature is
first fit to enforce source calibration (kappa(h_s)=1), matching the theorem's premise.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression


def sample_binary_csbm(N, p, q, mu, sigma, rng):
    """F=1 features for a clean match to the scalar theory."""
    y = rng.integers(0, 2, N) * 2 - 1                      # +/-1
    x = y * mu + sigma * rng.standard_normal(N)            # scalar feature
    same = (y[:, None] == y[None, :])
    Pmat = np.where(same, p, q)
    R = rng.random((N, N))
    U = np.triu(R < Pmat, k=1)
    A = (U | U.T).astype(float)
    return A, x, y


def mean_aggregate(A, x):
    d = A.sum(1)
    z = (A @ x) / np.maximum(d, 1.0)
    return z, d


def edge_homophily(A, y):
    iu = np.triu_indices_from(A, k=1)
    e = A[iu] > 0
    same = (y[:, None] == y[None, :])[iu]
    return float(same[e].mean()) if e.sum() else 0.0


def opt_temperature(logit, y01):
    """NLL-optimal temperature for binary logits; y01 in {0,1}."""
    def nll(logT):
        T = np.exp(logT)
        p = 1.0 / (1.0 + np.exp(-np.clip(logit / T, -30, 30)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.mean(y01 * np.log(p) + (1 - y01) * np.log(1 - p))
    r = minimize_scalar(nll, bounds=(-4, 4), method="bounded")
    return float(np.exp(r.x))


def ece_binary(prob_pos, y01, n_bins=15):
    conf = np.maximum(prob_pos, 1 - prob_pos)
    pred = (prob_pos >= 0.5).astype(int)
    correct = (pred == y01).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1]) if b else (conf >= bins[b]) & (conf <= bins[b + 1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def kappa_pred(ht, hs, rho):
    return ((2 * ht - 1) * (1 + 4 * hs * (1 - hs) * rho)) / \
           ((2 * hs - 1) * (1 + 4 * ht * (1 - ht) * rho))


def run(hs=0.8, rho=1.0, N=4000, deg=25, mu=1.0, ht_grid=None, seeds=6):
    sigma = mu / np.sqrt(rho)
    if ht_grid is None:
        ht_grid = np.array([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    ps = deg * 2.0 / N                                     # p+q, keeps degree ~const
    rows = []
    for ht in ht_grid:
        km, Tm, ece_un, ece_or, acc, hs_meas, ht_meas = [], [], [], [], [], [], []
        for s in range(seeds):
            rng = np.random.default_rng(1000 * s + int(1000 * ht))
            # --- source graph (homophily hs) ---
            p_s, q_s = ps * hs, ps * (1 - hs)
            A_s, x_s, y_s = sample_binary_csbm(N, p_s, q_s, mu, sigma, rng)
            z_s, _ = mean_aggregate(A_s, x_s)
            y01_s = (y_s > 0).astype(int)
            clf = LogisticRegression(C=1e6, solver="lbfgs")
            clf.fit(z_s.reshape(-1, 1), y01_s)
            logit_s = clf.decision_function(z_s.reshape(-1, 1))
            T_src = opt_temperature(logit_s, y01_s)          # enforce source calibration
            hs_meas.append(edge_homophily(A_s, y_s))
            # --- target graph (homophily ht) ---
            p_t, q_t = ps * ht, ps * (1 - ht)
            A_t, x_t, y_t = sample_binary_csbm(N, p_t, q_t, mu, sigma, rng)
            z_t, _ = mean_aggregate(A_t, x_t)
            y01_t = (y_t > 0).astype(int)
            logit_t = clf.decision_function(z_t.reshape(-1, 1)) / T_src
            T_or = opt_temperature(logit_t, y01_t)           # target-optimal temp
            km.append(1.0 / T_or); Tm.append(T_or)
            p_un = 1 / (1 + np.exp(-np.clip(logit_t, -30, 30)))
            p_or = 1 / (1 + np.exp(-np.clip(logit_t / T_or, -30, 30)))
            ece_un.append(ece_binary(p_un, y01_t))
            ece_or.append(ece_binary(p_or, y01_t))
            acc.append(((logit_t > 0).astype(int) == y01_t).mean())
            ht_meas.append(edge_homophily(A_t, y_t))
        rows.append(dict(
            ht=float(ht), ht_meas=float(np.mean(ht_meas)),
            kappa_pred=float(kappa_pred(np.mean(ht_meas), np.mean(hs_meas), rho)),
            kappa_meas=float(np.mean(km)), kappa_meas_sd=float(np.std(km)),
            T_pred=float(1 / kappa_pred(np.mean(ht_meas), np.mean(hs_meas), rho)),
            T_meas=float(np.mean(Tm)),
            acc=float(np.mean(acc)),
            ece_uncal=float(np.mean(ece_un)), ece_oracle=float(np.mean(ece_or))))
    return rows, float(np.mean(hs_meas)), sigma


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT = "figures"
    os.makedirs(OUT, exist_ok=True)

    HS, RHO = 0.8, 1.0
    rows, hs_meas, sigma = run(hs=HS, rho=RHO)
    print(f"source homophily (measured) = {hs_meas:.3f}, rho={RHO}, sigma={sigma:.3f}\n")
    print(f"{'h_t':>5} {'acc':>5} {'kappa_pred':>10} {'kappa_meas':>12} "
          f"{'T_pred':>7} {'T_meas':>7} {'ECE_unc':>8} {'ECE_or':>7}")
    for r in rows:
        print(f"{r['ht']:>5.2f} {r['acc']:>5.2f} {r['kappa_pred']:>10.3f} "
              f"{r['kappa_meas']:>7.3f}\u00b1{r['kappa_meas_sd']:.3f} "
              f"{r['T_pred']:>7.3f} {r['T_meas']:>7.3f} "
              f"{r['ece_uncal']:>8.4f} {r['ece_oracle']:>7.4f}")

    ht = [r["ht"] for r in rows]
    kp = [r["kappa_pred"] for r in rows]
    kmv = [r["kappa_meas"] for r in rows]
    ksd = [r["kappa_meas_sd"] for r in rows]
    ece = [r["ece_uncal"] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(ht, kp, "b-", lw=2, label=r"$\kappa$ predicted (closed form)")
    ax[0].errorbar(ht, kmv, yerr=ksd, fmt="ko", capsize=3,
                   label=r"$\kappa$ measured $=1/T_{oracle}$")
    ax[0].axhline(1, color="gray", ls="--", alpha=.7)
    ax[0].axvline(HS, color="orange", ls=":", alpha=.8, label="source homophily")
    ax[0].fill_between(ht, 0, 1, where=[k < 1 for k in kp], color="red", alpha=.06)
    ax[0].text(0.6, 0.3, "over-confident\n($\\kappa<1$)", color="#a00", fontsize=9)
    ax[0].text(0.86, 1.3, "under-\nconfident", color="#00a", fontsize=9)
    ax[0].set_xlabel("target homophily $h_t$"); ax[0].set_ylabel(r"calibration slope $\kappa$")
    ax[0].set_title(f"Theory matches simulation ($h_s$={HS}, $\\rho$={RHO})")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].plot(ht, ece, "k-o", label="uncalibrated ECE")
    ax[1].plot(ht, [r["ece_oracle"] for r in rows], "g--s", label="after $T^*=1/\\kappa$")
    ax[1].axvline(HS, color="orange", ls=":", alpha=.8)
    ax[1].set_xlabel("target homophily $h_t$"); ax[1].set_ylabel("ECE")
    ax[1].set_title("Miscalibration grows as $h_t$ leaves $h_s$")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_theory_validation.png", dpi=140)
    print("\nSaved: fig5_theory_validation.png")
