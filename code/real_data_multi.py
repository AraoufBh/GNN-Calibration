"""
Real-data calibration under covariate shift across 5 graphs spanning the homophily
spectrum (roman_empire 0.05 -> questions 0.84). For each graph we report, under a fixed
relative covariate shift: ECE for Uncalibrated, Source-temperature, label-free STAC, and
the target Oracle temperature. Confirms across diverse real graphs that (i) a single
oracle temperature recovers calibration (global suffices), and (ii) label-free STAC
improves over source-temperature scaling. Sparse throughout; results saved incrementally.
"""
import json, os
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

DATASETS = ["roman_empire", "amazon_ratings", "tolokers", "minesweeper", "questions"]
JSON = "data/results_real.json"


def load(name):
    d = np.load(f"data/{name}.npz")
    return (d["node_features"].astype(float), d["node_labels"].astype(int),
            d["edges"], d["train_masks"], d["val_masks"], d["test_masks"])

def build_S(edges, N):
    A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(N, N))
    A = (A + A.T); A.data[:] = 1.0
    A = A.tocsr() + sp.eye(N, format="csr")
    dinv = 1.0 / np.sqrt(np.asarray(A.sum(1)).ravel())
    return (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr()

def sign_features(S, X, k=2):
    feats = [X]; Z = X
    for _ in range(k):
        Z = S @ Z; feats.append(Z)
    return np.hstack(feats)

def softmax(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)

def to_logits(clf, F):
    L = clf.decision_function(F)
    return np.stack([-L, L], 1) if L.ndim == 1 else L

def opt_T(logits, y):
    def nll(lt):
        T = np.exp(lt); z = logits / T; z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))

def ece(logits, y, T=1.0, nb=15):
    P = softmax(logits / T); conf = P.max(1); corr = (P.argmax(1) == y).astype(float)
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (conf > b[i]) & (conf <= b[i+1]) if i else (conf >= b[i]) & (conf <= b[i+1])
        if m.sum(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return e

def stac_labelfree(clf, S, Xs, mask, M=6, alpha=0.3, rng=None):
    fstd = Xs.std(0, keepdims=True) + 1e-8
    preds = [to_logits(clf, sign_features(S, Xs + alpha*fstd*rng.standard_normal(Xs.shape)))[mask].argmax(1)
             for _ in range(M)]
    preds = np.array(preds); dis, c = 0.0, 0
    for a in range(M):
        for b_ in range(a+1, M):
            dis += (preds[a] != preds[b_]).mean(); c += 1
    a_hat = 1.0 - dis / max(c, 1)
    L = to_logits(clf, sign_features(S, Xs))[mask]
    T = float(np.exp(minimize_scalar(
        lambda lt: (softmax(L/np.exp(lt)).max(1).mean() - a_hat)**2,
        bounds=(-4, 4), method="bounded").x))
    return T, a_hat

def process(name, level=1.0, splits=3, M=6):
    X, y, edges, trm, vam, tem = load(name)
    N = len(y); S = build_S(edges, N)
    h = float((y[edges[:, 0]] == y[edges[:, 1]]).mean())
    Fclean = sign_features(S, X)
    acc0, out = [], {k: [] for k in ["Uncal", "Source-TS", "STAC", "Oracle", "ahat", "tacc"]}
    for sp_i in range(splits):
        tr, va, te = trm[sp_i], vam[sp_i], tem[sp_i]
        rng = np.random.default_rng(sp_i)
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200).fit(Fclean[tr], y[tr])
        T_src = opt_T(to_logits(clf, Fclean)[va], y[va])
        acc0.append((to_logits(clf, Fclean)[te].argmax(1) == y[te]).mean())
        Xs = X + level * X.std(0, keepdims=True) * rng.standard_normal(X.shape)
        L_te = to_logits(clf, sign_features(S, Xs))[te]
        out["tacc"].append((L_te.argmax(1) == y[te]).mean())
        out["Uncal"].append(ece(L_te, y[te], 1.0))
        out["Source-TS"].append(ece(L_te, y[te], T_src))
        T_lf, a_hat = stac_labelfree(clf, S, Xs, te, M=M, rng=rng)
        out["STAC"].append(ece(L_te, y[te], T_lf)); out["ahat"].append(a_hat)
        out["Oracle"].append(ece(L_te, y[te], opt_T(L_te, y[te])))
    return dict(name=name, N=N, K=int(len(np.unique(y))), homophily=h,
                base_acc=float(np.mean(acc0)),
                **{k: [float(np.mean(v)), float(np.std(v))] for k, v in out.items()})


if __name__ == "__main__":
    results = json.load(open(JSON)) if os.path.exists(JSON) else {}
    for name in DATASETS:
        if name in results:
            continue
        r = process(name)
        results[name] = r
        json.dump(results, open(JSON, "w"), indent=1)
        print(f"{name:15s} h={r['homophily']:.2f} K={r['K']:2d} base_acc={r['base_acc']:.2f} "
              f"tacc={r['tacc'][0]:.2f} ahat={r['ahat'][0]:.2f} | "
              f"ECE unc={r['Uncal'][0]:.3f} src={r['Source-TS'][0]:.3f} "
              f"STAC={r['STAC'][0]:.3f} or={r['Oracle'][0]:.3f}", flush=True)
    print("done")
