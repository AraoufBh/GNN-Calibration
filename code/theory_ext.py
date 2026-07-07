"""
Theory extensions + numerical validation.

(A) Self-loop, symmetric-normalized GCN (real GCN operator  D~^-1/2 (A+I) D~^-1/2):
    in the ~regular-degree limit this is a mean over (neighbours + self); since the
    self term is always same-class it shifts the effective homophily. Predicted slope
        kappa_SL(h_t) = A(h_t) B(h_s) / [ A(h_s) B(h_t) ],
        A(h)=1+d(2h-1),  B(h)=4 d h(1-h) rho + (d+1).
    (d->inf reduces to the mean-aggregation slope.)

(B) K classes (symmetric simplex means, sum_c mu_c = 0): the aggregation signal
    coefficient generalizes to
        c_K(h) = (hK - 1)/(K - 1)          (zero at h = 1/K; = 2h-1 for K=2).
    Homophily shift rescales the whole logit vector by c_K(h_t)/c_K(h_s), so a single
    global temperature stays optimal. We validate c_K(h) directly and check that one
    global temperature recovers calibration under shift for K>2.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

SIG_CLIP = lambda a: np.clip(a, -30, 30)


# ---------- graph / aggregation ----------
def sample_csbm_K(N, K, p, q, mu_vecs, sigma, rng):
    y = rng.integers(0, K, N)
    X = mu_vecs[y] + sigma * rng.standard_normal((N, mu_vecs.shape[1]))
    same = (y[:, None] == y[None, :])
    Pmat = np.where(same, p, q)
    R = rng.random((N, N))
    U = np.triu(R < Pmat, k=1)
    A = (U | U.T).astype(float)
    return A, X, y

def mean_agg(A, X):
    d = A.sum(1)
    return (A @ X) / np.maximum(d, 1.0)[:, None], d

def gcn_agg(A, X):
    N = A.shape[0]
    Ah = A + np.eye(N)
    d = Ah.sum(1)
    dinv = 1.0 / np.sqrt(d)
    S = (dinv[:, None] * Ah) * dinv[None, :]
    return S @ X, A.sum(1)

def edge_homophily(A, y):
    iu = np.triu_indices_from(A, k=1)
    e = A[iu] > 0
    same = (y[:, None] == y[None, :])[iu]
    return float(same[e].mean()) if e.sum() else 0.0

def simplex_means(K, mu):
    E = np.eye(K) - np.ones((K, K)) / K       # centered one-hot, sum_c = 0
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    return mu * E                              # (K,K), equal norm mu


# ---------- temperature / ECE (multiclass) ----------
def opt_T(logits, y):
    def nll(logT):
        T = np.exp(logT)
        z = logits / T
        z = z - z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    r = minimize_scalar(nll, bounds=(-4, 4), method="bounded")
    return float(np.exp(r.x))

def softmax(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)

def ece(logits, y, T=1.0, n_bins=15):
    P = softmax(logits / T)
    conf = P.max(1); pred = P.argmax(1); corr = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b+1]) if b else (conf >= bins[b]) & (conf <= bins[b+1])
        if m.sum(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return e


# ---------- (A) self-loop GCN calibration slope ----------
def kappa_SL(ht, hs, rho, d):
    A = lambda h: 1 + d * (2 * h - 1)
    B = lambda h: 4 * d * h * (1 - h) * rho + (d + 1)
    return A(ht) * B(hs) / (A(hs) * B(ht))

def kappa_mean(ht, hs, rho):
    return ((2*ht-1)*(1+4*hs*(1-hs)*rho)) / ((2*hs-1)*(1+4*ht*(1-ht)*rho))

def validate_selfloop(hs=0.8, rho=1.0, N=3000, deg=20, mu=1.0, seeds=5):
    sigma = mu / np.sqrt(rho); ps = deg * 2.0 / N
    ht_grid = np.array([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    out = []
    for ht in ht_grid:
        km, dmeas = [], []
        for s in range(seeds):
            rng = np.random.default_rng(7*s + int(1000*ht))
            mv = np.array([[mu], [-mu]])                      # binary, F=1 via +/-mu
            A_s, X_s, y_s = sample_csbm_K(N, 2, ps*hs, ps*(1-hs), mv, sigma, rng)
            z_s, d_s = gcn_agg(A_s, X_s)
            clf = LogisticRegression(C=1e6, solver="lbfgs").fit(z_s, y_s)
            L_s = clf.decision_function(z_s)
            L_s = np.stack([-L_s, L_s], 1)
            T_src = opt_T(L_s, y_s)
            A_t, X_t, y_t = sample_csbm_K(N, 2, ps*ht, ps*(1-ht), mv, sigma, rng)
            z_t, d_t = gcn_agg(A_t, X_t)
            L_t = clf.decision_function(z_t); L_t = np.stack([-L_t, L_t], 1) / T_src
            km.append(1.0 / opt_T(L_t, y_t)); dmeas.append(d_t.mean())
        d_bar = float(np.mean(dmeas))
        out.append((float(ht), kappa_SL(ht, hs, rho, d_bar),
                    kappa_mean(ht, hs, rho), float(np.mean(km)), float(np.std(km))))
    return out, sigma


# ---------- (B) K-class signal coefficient + global-T ----------
def c_K(h, K):
    return (h * K - 1) / (K - 1)

def validate_multiclass(Ks=(3, 4, 5), hs=0.8, rho=1.0, N=3000, deg=25, mu=1.0, seeds=4):
    sigma = mu / np.sqrt(rho); ps = deg * 2.0 / N
    hparam_grid = np.array([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    coeff = {}     # K -> list of (h_meas, c_pred_at_hmeas, c_meas)
    ecetab = {}    # K -> list of (h_meas, ece_uncal, ece_globalT)
    for K in Ks:
        mv = simplex_means(K, mu)
        coeff[K] = []; ecetab[K] = []
        for hp in hparam_grid:
            cs, eu, eg, hm = [], [], [], []
            for s in range(seeds):
                rng = np.random.default_rng(11*s + int(1000*hp) + 100*K)
                A_s, X_s, y_s = sample_csbm_K(N, K, ps*hs, ps*(1-hs), mv, sigma, rng)
                z_s, _ = mean_agg(A_s, X_s)
                clf = LogisticRegression(C=1e6, solver="lbfgs",
                                         max_iter=500).fit(z_s, y_s)
                L_s = clf.decision_function(z_s); T_src = opt_T(L_s, y_s)
                A_t, X_t, y_t = sample_csbm_K(N, K, ps*hp, ps*(1-hp), mv, sigma, rng)
                z_t, _ = mean_agg(A_t, X_t)
                hm.append(edge_homophily(A_t, y_t))     # ACTUAL edge homophily
                proj = (z_t * mv[y_t]).sum(1) / (mu ** 2)
                cs.append(proj.mean())
                L_t = clf.decision_function(z_t) / T_src
                T_or = opt_T(L_t, y_t)
                eu.append(ece(L_t, y_t, 1.0)); eg.append(ece(L_t, y_t, T_or))
            h_meas = float(np.mean(hm))
            coeff[K].append((h_meas, c_K(h_meas, K), float(np.mean(cs))))
            ecetab[K].append((h_meas, float(np.mean(eu)), float(np.mean(eg))))
    return coeff, ecetab


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT = "figures"

    # ---- (A) self-loop GCN ----
    HS, RHO = 0.8, 1.0
    sl, sigma = validate_selfloop(hs=HS, rho=RHO)
    print("=== (A) self-loop GCN: kappa_SL vs measured ===")
    print(f"{'h_t':>5} {'kSL_pred':>9} {'kmean':>7} {'k_meas':>13}")
    for ht, ksl, km, kmv, ksd in sl:
        print(f"{ht:>5.2f} {ksl:>9.3f} {km:>7.3f} {kmv:>7.3f}\u00b1{ksd:.3f}")

    # ---- (B) K classes ----
    coeff, ecetab = validate_multiclass(hs=HS, rho=RHO)
    print("\n=== (B) K-class signal coefficient c_K(h): predicted vs measured ===")
    for K in coeff:
        print(f" K={K}:  " + "  ".join(
            f"h{ht:.2f}[{cp:+.2f}/{cm:+.2f}]" for ht, cp, cm in coeff[K]))
    print("\n=== (B) single global temperature recovers calibration (K=3) ===")
    print(f"{'h_t':>5} {'ECE_uncal':>10} {'ECE_globalT':>12}")
    for ht, eu, eg in ecetab[3]:
        print(f"{ht:>5.2f} {eu:>10.4f} {eg:>12.4f}")

    # ---- figures ----
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ht = [r[0] for r in sl]
    ax[0].plot(ht, [r[1] for r in sl], "b-", lw=2, label=r"$\kappa_{SL}$ predicted (self-loop)")
    ax[0].plot(ht, [r[2] for r in sl], "c--", lw=1.5, label=r"$\kappa$ mean-agg (no self-loop)")
    ax[0].errorbar(ht, [r[3] for r in sl], yerr=[r[4] for r in sl], fmt="ko",
                   capsize=3, label=r"$\kappa$ measured (GCN sim)")
    ax[0].axhline(1, color="gray", ls="--", alpha=.6); ax[0].axvline(HS, color="orange", ls=":")
    ax[0].set_xlabel("target homophily $h_t$"); ax[0].set_ylabel(r"calibration slope $\kappa$")
    ax[0].set_title("(A) Self-loop GCN: theory matches sim"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    colors = {3: "#1f77b4", 4: "#ff7f0e", 5: "#2ca02c"}
    hgrid = np.linspace(0.3, 0.85, 100)
    for K in coeff:
        ax[1].plot(hgrid, c_K(hgrid, K), "-", color=colors[K], label=f"theory K={K}")
        ax[1].plot([r[0] for r in coeff[K]], [r[2] for r in coeff[K]], "o",
                   color=colors[K], ms=5)
        ax[1].axvline(1.0 / K, color=colors[K], ls=":", alpha=.4)
    ax[1].axhline(0, color="gray", ls="--", alpha=.5)
    ax[1].set_xlabel("edge homophily $h$ (measured)")
    ax[1].set_ylabel(r"signal coefficient $c_K(h)$")
    ax[1].set_title(r"(B) $c_K(h)=(hK{-}1)/(K{-}1)$: lines=theory, dots=sim")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_theory_extensions.png", dpi=140)
    print("\nSaved: fig6_theory_extensions.png")
