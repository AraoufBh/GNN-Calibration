"""
Experiment B: covariate-shift strength sweep on real graphs.
Proposition 4 predicts covariate shift makes the model progressively OVER-confident
(mean confidence > accuracy, growing with the shift strength gamma), and that a single
oracle temperature restores calibration. We verify both on 5 real graphs.
"""
import json, os
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

DATASETS = ["roman_empire", "amazon_ratings", "tolokers", "minesweeper", "questions"]
LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
JSON = "data/results_shift_sweep.json"


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
        z = logits / np.exp(lt); z -= z.max(1, keepdims=True)
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


def process(name, splits=2):
    X, y, edges, trm, vam, tem = load(name)
    N = len(y); S = build_S(edges, N); Fclean = sign_features(S, X)
    h = float((y[edges[:, 0]] == y[edges[:, 1]]).mean())
    out = {k: {m: [] for m in ["unc", "src", "orc", "conf", "acc"]} for k in map(str, LEVELS)}
    for sp_i in range(splits):
        tr, va, te = trm[sp_i], vam[sp_i], tem[sp_i]
        rng = np.random.default_rng(sp_i)
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200).fit(Fclean[tr], y[tr])
        Tsrc = opt_T(to_logits(clf, Fclean)[va], y[va])
        std = X.std(0, keepdims=True)
        for lv in LEVELS:
            Xs = X + lv * std * rng.standard_normal(X.shape)
            L = to_logits(clf, sign_features(S, Xs))[te]
            P = softmax(L)
            out[str(lv)]["unc"].append(ece(L, y[te], 1.0))
            out[str(lv)]["src"].append(ece(L, y[te], Tsrc))
            out[str(lv)]["orc"].append(ece(L, y[te], opt_T(L, y[te])))
            out[str(lv)]["conf"].append(float(P.max(1).mean()))
            out[str(lv)]["acc"].append(float((P.argmax(1) == y[te]).mean()))
    agg = {lv: {m: float(np.mean(v)) for m, v in d.items()} for lv, d in out.items()}
    return dict(name=name, homophily=h, K=int(len(np.unique(y))), levels=LEVELS, curve=agg)


if __name__ == "__main__":
    res = json.load(open(JSON)) if os.path.exists(JSON) else {}
    for name in DATASETS:
        if name in res:
            continue
        r = process(name); res[name] = r
        json.dump(res, open(JSON, "w"), indent=1)
        gaps = [r["curve"][str(lv)]["conf"] - r["curve"][str(lv)]["acc"] for lv in LEVELS]
        print(f"{name:15s} h={r['homophily']:.2f} conf-acc gap by level: "
              + " ".join(f"{g:+.3f}" for g in gaps), flush=True)
    print("done")
