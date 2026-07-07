"""
Real-data check (minesweeper heterophilous graph, 10k nodes) of the paper's
PRACTICAL claims under covariate shift:
  - a frozen linear-GNN becomes mis-calibrated under shift,
  - a source-fit temperature does NOT fix it,
  - a single label-free temperature (STAC-global) recovers calibration, approaching
    the target-oracle temperature  (=> global suffices, matching the theory).
Sparse operations throughout; linear SIGN-style features [X, SX, S^2 X] + logistic reg.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

d = np.load("data/minesweeper.npz")
X0 = d["node_features"].astype(float)
y = d["node_labels"].astype(int)
edges = d["edges"]
N, K = len(y), len(np.unique(y))

# symmetric normalized adjacency with self-loops (sparse)
A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(N, N))
A = (A + A.T)
A.data[:] = 1.0
A = A.tocsr() + sp.eye(N, format="csr")
deg = np.asarray(A.sum(1)).ravel()
Dinv = sp.diags(1.0 / np.sqrt(deg))
S = (Dinv @ A @ Dinv).tocsr()

# edge homophily of the real graph
ii, jj = edges[:, 0], edges[:, 1]
h_real = float((y[ii] == y[jj]).mean())


def sign_features(Xraw, k=2):
    feats = [Xraw]; Z = Xraw
    for _ in range(k):
        Z = S @ Z; feats.append(Z)
    return np.hstack(feats)


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)

def to_logits(clf, F):
    L = clf.decision_function(F)
    return np.stack([-L, L], 1) if L.ndim == 1 else L

def opt_T(logits, yy):
    def nll(lt):
        T = np.exp(lt); z = logits / T; z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        return -np.log(np.clip(p[np.arange(len(yy)), yy], 1e-12, 1)).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))

def ece(logits, yy, T=1.0, nb=15):
    P = softmax(logits / T); conf = P.max(1); corr = (P.argmax(1) == yy).astype(float)
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (conf > b[i]) & (conf <= b[i+1]) if i else (conf >= b[i]) & (conf <= b[i+1])
        if m.sum(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return e


def stac_global_labelfree(clf, Xshift, mask, M=10, alpha=0.3, rng=None):
    """Label-free single temperature: estimate target accuracy via feature-perturbation
    disagreement (GDE), then pick T so mean confidence == estimated accuracy."""
    fstd = Xshift.std(0, keepdims=True) + 1e-8
    preds = []
    for _ in range(M):
        Fp = sign_features(Xshift + alpha * fstd * rng.standard_normal(Xshift.shape))
        preds.append(to_logits(clf, Fp)[mask].argmax(1))
    preds = np.array(preds)
    # GDE error estimate = mean pairwise disagreement
    dis, c = 0.0, 0
    for a in range(M):
        for b_ in range(a + 1, M):
            dis += (preds[a] != preds[b_]).mean(); c += 1
    a_hat = 1.0 - dis / max(c, 1)
    L = to_logits(clf, sign_features(Xshift))[mask]
    def obj(lt):
        T = np.exp(lt); return (softmax(L / T).max(1).mean() - a_hat) ** 2
    T = float(np.exp(minimize_scalar(obj, bounds=(-4, 4), method="bounded").x))
    return T, a_hat


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT = "figures"

    levels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    res = {k: [] for k in ["Uncal", "Source-TS", "STAC-global", "Oracle-TS", "acc", "ahat", "tgt_acc"]}

    for lv in levels:
        agg = {k: [] for k in res}
        for split in range(5):
            tr = d["train_masks"][split]; va = d["val_masks"][split]; te = d["test_masks"][split]
            rng = np.random.default_rng(split)
            Fclean = sign_features(X0)
            clf = LogisticRegression(C=1.0, max_iter=500).fit(Fclean[tr], y[tr])
            # source temperature on clean val
            T_src = opt_T(to_logits(clf, Fclean)[va], y[va])
            # target = covariate-shifted features
            Xs = X0 + lv * rng.standard_normal(X0.shape)
            L_te = to_logits(clf, sign_features(Xs))[te]
            agg["acc"].append((L_te.argmax(1) == y[te]).mean())
            agg["tgt_acc"].append((L_te.argmax(1) == y[te]).mean())
            agg["Uncal"].append(ece(L_te, y[te], 1.0))
            agg["Source-TS"].append(ece(L_te, y[te], T_src))
            T_lf, a_hat = stac_global_labelfree(clf, Xs, te, rng=rng)
            agg["STAC-global"].append(ece(L_te, y[te], T_lf))
            agg["ahat"].append(a_hat)
            T_or = opt_T(L_te, y[te])
            agg["Oracle-TS"].append(ece(L_te, y[te], T_or))
        for k in res:
            res[k].append((np.mean(agg[k]), np.std(agg[k])))

    print(f"minesweeper: N={N}, edges={len(edges)}, K={K}, edge-homophily={h_real:.3f}\n")
    print(f"{'noise':>6} {'tgt_acc':>7} {'ahat':>6} {'ECE_unc':>8} {'Src-TS':>8} "
          f"{'STAC-g':>8} {'Oracle':>8}")
    for i, lv in enumerate(levels):
        print(f"{lv:>6.1f} {res['tgt_acc'][i][0]:>7.3f} {res['ahat'][i][0]:>6.3f} "
              f"{res['Uncal'][i][0]:>8.4f} {res['Source-TS'][i][0]:>8.4f} "
              f"{res['STAC-global'][i][0]:>8.4f} {res['Oracle-TS'][i][0]:>8.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    cols = {"Uncal": "#888", "Source-TS": "#d62728", "STAC-global": "#1f77b4", "Oracle-TS": "#2ca02c"}
    for m in ["Uncal", "Source-TS", "STAC-global", "Oracle-TS"]:
        mu = [res[m][i][0] for i in range(len(levels))]
        sd = [res[m][i][1] for i in range(len(levels))]
        ax.errorbar(levels, mu, yerr=sd, marker="o", capsize=3, label=m, color=cols[m])
    ax.set_xlabel("covariate-shift intensity"); ax.set_ylabel("ECE (\u2193)")
    ax.set_title(f"Real graph (minesweeper, h={h_real:.2f}): label-free STAC recovers calibration")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig8_real_minesweeper.png", dpi=140)
    print("\nSaved: fig8_real_minesweeper.png")
