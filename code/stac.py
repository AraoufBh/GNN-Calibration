"""
STAC: Structure-aware Test-time Adaptive Calibration of GNNs under distribution shift.

Setting (source-free graph domain adaptation for CALIBRATION, not accuracy):
  - A GCN is trained on a SOURCE graph G_s (labels available).
  - At test time we get a shifted TARGET graph G_t (same feature/label space,
    different distribution). NO target labels, NO source data are available.
  - Goal: recalibrate the frozen GCN's confidences on G_t.

Key idea (the novel bit):
  On a graph, structure gives a LABEL-FREE correctness signal.
  We use predictive instability under edge-dropout as a proxy of "P(correct)",
  and homophily-corrected neighbour agreement + logit signals as per-node features.
  We fit a tiny linear map (~6 params) from these signals to a PER-NODE temperature,
  by minimizing a label-free (binned) calibration objective on the target graph.

Everything is NumPy + SciPy (no GPU, no torch) -> lightweight & fully reproducible.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def onehot(y, C):
    O = np.zeros((y.shape[0], C))
    O[np.arange(y.shape[0]), y] = 1.0
    return O

def norm_adj(A):
    """Symmetric normalized adjacency with self-loops: D^-1/2 (A+I) D^-1/2."""
    N = A.shape[0]
    Ah = A + np.eye(N)
    d = Ah.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return (dinv[:, None] * Ah) * dinv[None, :]

# ----------------------------------------------------------------------------
# CSBM data with controllable homophily + shift
# ----------------------------------------------------------------------------

def sample_csbm(N, K, F, p_in, p_out, sep, feat_noise, rng, M):
    """Contextual SBM with GIVEN class-mean directions M (KxF).
    Sharing M across source/target keeps the feature-label mapping fixed."""
    y = rng.integers(0, K, size=N)
    X = sep * M[y] + feat_noise * rng.standard_normal((N, F))
    # edges
    same = (y[:, None] == y[None, :])
    P = np.where(same, p_in, p_out)
    R = rng.random((N, N))
    U = np.triu(R < P, k=1)
    A = (U | U.T).astype(np.float64)
    return A, X, y

def edge_homophily(A, y):
    iu = np.triu_indices_from(A, k=1)
    e = A[iu] > 0
    if e.sum() == 0:
        return 0.0
    same = (y[:, None] == y[None, :])[iu]
    return float(same[e].mean())

# ----------------------------------------------------------------------------
# 2-layer GCN in NumPy (hand-coded backprop + Adam)
# ----------------------------------------------------------------------------

class GCN:
    def __init__(self, F, H, C, rng):
        def glorot(fan_in, fan_out):
            s = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-s, s, size=(fan_in, fan_out))
        self.W1 = glorot(F, H)
        self.W2 = glorot(H, C)

    def forward(self, S, X, cache=False):
        A1 = S @ X
        Z1 = A1 @ self.W1
        H1 = np.maximum(Z1, 0.0)
        A2 = S @ H1
        Z2 = A2 @ self.W2  # logits
        if cache:
            self._c = (A1, Z1, H1, A2)
        return Z2

    def train(self, S, X, y, train_mask, val_mask, C,
              epochs=150, lr=0.01, wd=5e-4, verbose=False):
        # Adam state
        mW1 = np.zeros_like(self.W1); vW1 = np.zeros_like(self.W1)
        mW2 = np.zeros_like(self.W2); vW2 = np.zeros_like(self.W2)
        b1, b2, eps = 0.9, 0.999, 1e-8
        ntr = train_mask.sum()
        for t in range(1, epochs + 1):
            Z2 = self.forward(S, X, cache=True)
            A1, Z1, H1, A2 = self._c
            P = softmax(Z2)
            G = P - onehot(y, C)
            G[~train_mask] = 0.0
            G /= ntr
            dW2 = A2.T @ G + 2 * wd * self.W2
            dA2 = G @ self.W2.T
            dH1 = S.T @ dA2
            dZ1 = dH1 * (Z1 > 0)
            dW1 = A1.T @ dZ1 + 2 * wd * self.W1
            # Adam
            for (Wp, dW, m, v) in [(self.W1, dW1, mW1, vW1),
                                   (self.W2, dW2, mW2, vW2)]:
                m *= b1; m += (1 - b1) * dW
                v *= b2; v += (1 - b2) * (dW * dW)
                mhat = m / (1 - b1 ** t)
                vhat = v / (1 - b2 ** t)
                Wp -= lr * mhat / (np.sqrt(vhat) + eps)
            if verbose and (t % 50 == 0 or t == 1):
                pred = Z2.argmax(1)
                tra = (pred[train_mask] == y[train_mask]).mean()
                va = (pred[val_mask] == y[val_mask]).mean()
                print(f"  epoch {t:3d}  train_acc={tra:.3f}  val_acc={va:.3f}")

# ----------------------------------------------------------------------------
# Calibration metrics
# ----------------------------------------------------------------------------

def ece_mce(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; mce = 0.0; N = len(conf)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc = correct[m].mean(); cf = conf[m].mean(); w = m.mean()
        gap = abs(acc - cf)
        ece += w * gap
        mce = max(mce, gap)
    return ece, mce

def reliability(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    mids, accs, confs, ws = [], [], [], []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        mids.append((lo + hi) / 2)
        if m.sum() == 0:
            accs.append(np.nan); confs.append(np.nan); ws.append(0.0)
        else:
            accs.append(correct[m].mean()); confs.append(conf[m].mean()); ws.append(m.mean())
    return np.array(mids), np.array(accs), np.array(confs), np.array(ws)

def metrics(P, y):
    C = P.shape[1]
    pred = P.argmax(1)
    correct = (pred == y).astype(float)
    conf = P.max(1)
    ece, mce = ece_mce(conf, correct)
    brier = float(((P - onehot(y, C)) ** 2).sum(1).mean())
    nll = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())
    err = 1.0 - correct
    # error-detection AUROC: rank nodes by (1-conf); can only improve with PER-NODE T
    if 0 < err.sum() < len(err):
        auroc = float(roc_auc_score(err, 1.0 - conf))
    else:
        auroc = float("nan")
    # selective risk at 80% coverage: error rate on the 80% most-confident nodes
    k = max(1, int(0.8 * len(conf)))
    keep = np.argsort(-conf)[:k]
    risk80 = float(err[keep].mean())
    return dict(acc=float(correct.mean()), ece=ece, mce=mce, brier=brier,
                nll=nll, auroc=auroc, risk80=risk80)

# ----------------------------------------------------------------------------
# Temperature scaling helpers
# ----------------------------------------------------------------------------

def apply_temp(logits, T):
    T = np.asarray(T, dtype=float)
    if T.ndim == 0:
        return softmax(logits / T)
    return softmax(logits / T[:, None])

def mean_feat_instability(gcn, S, X, mask, alpha, M, rng):
    """Mean prediction flip-rate under feature perturbation over masked nodes."""
    base = gcn.forward(S, X).argmax(1)
    fsd = X.std(0, keepdims=True) + 1e-8
    dis = np.zeros(X.shape[0])
    for _ in range(M):
        pred = gcn.forward(S, X + alpha * fsd * rng.standard_normal(X.shape)).argmax(1)
        dis += (pred != base)
    return float((dis / M)[mask].mean())

def calibrate_alpha(gcn, S, X, mask, target_err, M, rng, lo=0.02, hi=2.5, iters=8):
    """Bisection: find alpha s.t. mean feature-instability on source-val = source error."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        err_mid = mean_feat_instability(gcn, S, X, mask, mid, M, rng)
        if err_mid < target_err:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def atc_threshold(scores_src, acc_src):
    """ATC (Garg et al. 2022): threshold t s.t. frac(scores<t)=error_rate on source-val."""
    return float(np.quantile(scores_src, max(1e-3, 1.0 - acc_src)))

def atc_estimate(scores_tgt, t):
    """Label-free target-accuracy estimate = frac of target scores above threshold."""
    return float((scores_tgt >= t).mean())

def fit_global_T_nll(logits, y):
    """Standard temperature scaling: single T minimizing NLL on labelled data."""
    def obj(logt):
        T = np.exp(logt[0])
        P = apply_temp(logits, T)
        return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean()
    res = minimize(obj, x0=[0.0], method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-6, maxiter=500))
    return float(np.exp(res.x[0]))

# ----------------------------------------------------------------------------
# STAC: structural signals + label-free recalibration
# ----------------------------------------------------------------------------

def structural_signals(gcn, A, X, base_logits, M=10, p_drop=0.3,
                       feat_alpha=0.15, rng=None):
    """Return dict of per-node structural / logit signals on the target graph."""
    N = A.shape[0]
    S = norm_adj(A)
    P = softmax(base_logits)
    base_pred = base_logits.argmax(1)
    K = base_logits.shape[1]

    # 1a) predictive instability under FEATURE perturbation (probes local margin)
    feat_sd = X.std(0, keepdims=True) + 1e-8
    dis_feat = np.zeros(N); preds_feat = []
    for _ in range(M):
        Xp = X + feat_alpha * feat_sd * rng.standard_normal(X.shape)
        pred_p = gcn.forward(S, Xp).argmax(1)
        preds_feat.append(pred_p)
        dis_feat += (pred_p != base_pred)
    inst_feat = dis_feat / M
    preds_feat = np.array(preds_feat)

    # 1b) predictive instability under EDGE-dropout (probes structural ambiguity)
    iu = np.triu_indices(N, k=1)
    edge_mask = A[iu] > 0
    disagree = np.zeros(N); preds_edge = []
    for _ in range(M):
        keep = rng.random(edge_mask.sum()) > p_drop
        vals = np.zeros(edge_mask.sum()); vals[keep] = 1.0
        Ad = np.zeros((N, N))
        rows, cols = iu[0][edge_mask], iu[1][edge_mask]
        Ad[rows, cols] = vals
        Ad = Ad + Ad.T
        Sd = norm_adj(Ad)
        pred_d = gcn.forward(Sd, X).argmax(1)
        preds_edge.append(pred_d)
        disagree += (pred_d != base_pred)
    inst_edge = disagree / M
    preds_edge = np.array(preds_edge)

    # GDE-style label-free ERROR estimate = mean pairwise disagreement across passes
    def pairwise_dis(preds):
        tot, cnt = 0.0, 0
        for i in range(preds.shape[0]):
            for j in range(i + 1, preds.shape[0]):
                tot += (preds[i] != preds[j]).mean(); cnt += 1
        return tot / max(cnt, 1)
    gde_feat = 1.0 - pairwise_dis(preds_feat)
    gde_edge = 1.0 - pairwise_dis(preds_edge)

    # 2) neighbour pseudo-label agreement + estimated homophily
    deg = A.sum(1)
    same_pred = (base_pred[:, None] == base_pred[None, :]).astype(float)
    agree = (A * same_pred).sum(1) / np.maximum(deg, 1)   # frac neighbours w/ same pred
    agree[deg == 0] = 0.5
    # estimated edge homophily from pseudo-labels (label-free)
    e = A[iu] > 0
    h_hat = float(same_pred[iu][e].mean()) if e.sum() > 0 else 0.5
    # homophily-corrected agreement: high => consistent with graph's pattern
    agree_corr = agree if h_hat >= 0.5 else 1.0 - agree

    # 3) logit-only signals
    entropy = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1) / np.log(K)   # in [0,1]
    sortP = np.sort(P, axis=1)
    margin = sortP[:, -1] - sortP[:, -2]
    logdeg = np.log(deg + 1.0)

    return dict(inst_feat=inst_feat, inst_edge=inst_edge, agree_corr=agree_corr,
                entropy=entropy, margin=margin, logdeg=logdeg, h_hat=h_hat,
                gde_feat=gde_feat, gde_edge=gde_edge)

def _standardize(cols):
    Phi = np.stack(cols, axis=1).astype(float)
    mu = Phi.mean(0); sd = Phi.std(0) + 1e-8
    return (Phi - mu) / sd

def _temp_from_params(Phi, params, Tmin=0.05):
    lin = Phi @ params[:-1] + params[-1]
    return Tmin + np.log1p(np.exp(np.clip(lin, -30, 30)))   # softplus

def _binned_cal_loss(conf, q, n_bins=15):
    """Label-free calibration loss: match confidence to correctness-proxy q per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    loss = 0.0; N = len(conf)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        loss += m.mean() * (conf[m].mean() - q[m].mean()) ** 2
    return loss

def stac_global(logits, proxy, restarts=4, rng=None):
    """Stage 1: single temperature pinning mean-confidence to the label-free level."""
    def obj(logt):
        T = np.exp(logt[0])
        conf = apply_temp(logits, T).max(1)
        return _binned_cal_loss(conf, proxy) + 2.0 * (conf.mean() - proxy.mean()) ** 2
    best = None
    for _ in range(restarts):
        r = minimize(obj, x0=[rng.uniform(-1, 1)], method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-8, maxiter=400))
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.x[0]))

def stac_pernode(logits, sig, proxy, T_glob, feature_set="full",
                 c=0.5, lam=1e-2, restarts=3, rng=None):
    """Stage 2: bounded per-node modulation T_i = T_glob * exp(c*tanh(phi.w+b))."""
    if feature_set == "full":
        cols = [sig["inst_feat"], sig["inst_edge"], sig["agree_corr"],
                sig["entropy"], sig["margin"], sig["logdeg"]]
    elif feature_set == "nostruct":
        cols = [sig["entropy"], sig["margin"]]     # logits only, no graph structure
    else:
        raise ValueError(feature_set)
    Phi = _standardize(cols)
    d = Phi.shape[1]

    def temp(params):
        delta = np.tanh(Phi @ params[:-1] + params[-1])
        return T_glob * np.exp(c * delta)

    def obj(params):
        T = temp(params)
        conf = apply_temp(logits, T).max(1)
        return (_binned_cal_loss(conf, proxy) + 2.0 * (conf.mean() - proxy.mean()) ** 2
                + lam * np.sum(params[:-1] ** 2))

    best = None
    for _ in range(restarts):
        x0 = np.concatenate([0.05 * rng.standard_normal(d), [0.0]])
        r = minimize(obj, x0=x0, method="Powell",
                     options=dict(xtol=1e-4, ftol=1e-8, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    return apply_temp(logits, temp(best.x))

# ----------------------------------------------------------------------------
# One full experiment run
# ----------------------------------------------------------------------------

def run_once(seed, shift_kind, shift_level,
             N=1000, K=3, F=16, H=32, sep=1.0, base_noise=2.5,
             epochs=120, M=8, verbose=False):
    rng = np.random.default_rng(seed)

    # shared class-mean directions (fixed feature-label mapping for source & target)
    Mu = rng.standard_normal((K, F))
    Mu = Mu / np.linalg.norm(Mu, axis=1, keepdims=True)

    # ---- source graph (homophilic) ----
    p_in_s, p_out_s = 0.020, 0.002
    A_s, X_s, y_s = sample_csbm(N, K, F, p_in_s, p_out_s, sep, base_noise, rng, Mu)
    S_s = norm_adj(A_s)
    idx = rng.permutation(N)
    tr = np.zeros(N, bool); va = np.zeros(N, bool)
    tr[idx[:N // 2]] = True
    va[idx[N // 2:N // 2 + N // 4]] = True

    gcn = GCN(F, H, K, rng)
    gcn.train(S_s, X_s, y_s, tr, va, K, epochs=epochs, verbose=verbose)
    src_logits = gcn.forward(S_s, X_s)
    src_val_acc = float((src_logits[va].argmax(1) == y_s[va]).mean())
    T_src = fit_global_T_nll(src_logits[va], y_s[va])             # source-fit temp
    # ATC threshold (scalar) stored from source val for label-free accuracy estimation
    t_atc = atc_threshold(softmax(src_logits[va]).max(1), src_val_acc)

    # ---- target graph (shifted) ----
    if shift_kind == "feature":
        p_in_t, p_out_t = p_in_s, p_out_s
        A_t, X_t, y_t = sample_csbm(N, K, F, p_in_t, p_out_t, sep, base_noise, rng, Mu)
        X_t = X_t + shift_level * rng.standard_normal(X_t.shape)   # covariate shift
    elif shift_kind == "homophily":
        # shift_level in [0,1]: 0 -> homophilic like source, 1 -> heterophilic
        p_out_t = 0.002 + shift_level * 0.010
        p_in_t = 0.020 - shift_level * 0.012
        A_t, X_t, y_t = sample_csbm(N, K, F, p_in_t, p_out_t, sep, base_noise, rng, Mu)
    elif shift_kind == "mixed":
        # HETEROGENEOUS: only a random half of nodes gets a strong feature shift.
        # No single global temperature can calibrate both halves -> per-node needed.
        A_t, X_t, y_t = sample_csbm(N, K, F, p_in_s, p_out_s, sep, base_noise, rng, Mu)
        shifted = rng.random(N) < 0.5
        X_t[shifted] += shift_level * rng.standard_normal((int(shifted.sum()), F))
    else:
        raise ValueError(shift_kind)

    S_t = norm_adj(A_t)
    logits_t = gcn.forward(S_t, X_t)
    h_t = edge_homophily(A_t, y_t)

    # ---- signals & recalibration (source-free, label-free) ----
    sig = structural_signals(gcn, A_t, X_t, logits_t, M=M, p_drop=0.3,
                             feat_alpha=0.15, rng=rng)
    # label-free correctness proxy: RANK from perturbation stability, LEVEL from a
    # conservative combination of ATC and edge-perturbation disagreement (GDE).
    a_atc = atc_estimate(softmax(logits_t).max(1), t_atc)
    a_hat = min(a_atc, sig["gde_edge"])
    proxy_raw = 1.0 - np.clip(0.7 * sig["inst_feat"] + 0.3 * sig["inst_edge"], 0, 1)
    proxy = np.clip(proxy_raw - proxy_raw.mean() + a_hat, 1e-3, 1 - 1e-3)

    out = {}
    out["Uncal"]      = metrics(softmax(logits_t), y_t)
    out["Source-TS"]  = metrics(apply_temp(logits_t, T_src), y_t)
    T_glob = stac_global(logits_t, proxy, rng=rng)
    out["STAC-global"] = metrics(apply_temp(logits_t, T_glob), y_t)
    out["STAC-noStruct"] = metrics(
        stac_pernode(logits_t, sig, proxy, T_glob, "nostruct", rng=rng), y_t)
    P_f = stac_pernode(logits_t, sig, proxy, T_glob, "full", rng=rng)
    out["STAC-full"]  = metrics(P_f, y_t)
    T_or = fit_global_T_nll(logits_t, y_t)              # oracle (uses target labels)
    out["Oracle-TS"]  = metrics(apply_temp(logits_t, T_or), y_t)

    # diagnostics: do the label-free proxies correlate with true error?
    err = (logits_t.argmax(1) != y_t).astype(float)
    def corr(a):
        if a.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, err)[0, 1])
    diag = dict(corr_feat=corr(sig["inst_feat"]), corr_edge=corr(sig["inst_edge"]),
                src_val_acc=src_val_acc, tgt_acc=out["Uncal"]["acc"],
                a_atc=a_atc, a_gde_feat=sig["gde_feat"], a_gde_edge=sig["gde_edge"],
                a_hat=a_hat, T_src=T_src, T_oracle=T_or)

    return dict(metrics=out, h_target=h_t, diag=diag,
                P_reps={"Uncal": softmax(logits_t),
                        "Source-TS": apply_temp(logits_t, T_src),
                        "STAC-full": P_f},
                y=y_t)


def _print_run(r):
    d = r["diag"]
    print(f"tgt_acc={d['tgt_acc']:.3f} a_hat={d['a_hat']:.3f} T_or={d['T_oracle']:.2f}")
    for k, m in r["metrics"].items():
        print(f"{k:14s} ECE={m['ece']:.4f} MCE={m['mce']:.4f} "
              f"Brier={m['brier']:.4f} NLL={m['nll']:.4f}")


# ----------------------------------------------------------------------------
# Full experiment: multi-seed sweeps, aggregation, figures, tables
# ----------------------------------------------------------------------------

def aggregate(runs, methods, metric):
    """runs: list of metrics-dicts -> (mean, std) per method for a metric."""
    out = {}
    for mth in methods:
        vals = np.array([r[mth][metric] for r in runs])
        out[mth] = (float(vals.mean()), float(vals.std()))
    return out


if __name__ == "__main__":
    import os, csv, json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT = "figures"
    os.makedirs(OUT, exist_ok=True)
    METHODS = ["Uncal", "Source-TS", "STAC-global", "STAC-noStruct",
               "STAC-full", "Oracle-TS"]
    SEEDS = list(range(4))
    rng_cfg = dict(sep=1.0, base_noise=2.5, epochs=120, M=8, N=1000)

    feat_levels = [0.2, 0.5, 0.8, 1.1, 1.4]
    homo_levels = [0.15, 0.25, 0.35]
    mixed_levels = [0.8, 1.2, 1.6]

    data = {"feature": {}, "homophily": {}, "mixed": {}}
    diags = {"feature": {}, "homophily": {}, "mixed": {}}
    rep_run = None                              # a representative run for reliability plot

    def do(kind, levels):
        for lv in levels:
            runs, dg = [], []
            for s in SEEDS:
                r = run_once(seed=s, shift_kind=kind, shift_level=lv, **rng_cfg)
                runs.append(r["metrics"]); dg.append(r["diag"])
                global rep_run
                if kind == "feature" and abs(lv - 0.8) < 1e-9 and s == 0:
                    rep_run = r
            data[kind][lv] = runs; diags[kind][lv] = dg
            ece = aggregate(runs, METHODS, "ece")
            print(f"[{kind} {lv}] tgt_acc={np.mean([d['tgt_acc'] for d in dg]):.3f} "
                  f"a_hat={np.mean([d['a_hat'] for d in dg]):.3f} | "
                  f"ECE Src={ece['Source-TS'][0]:.3f} "
                  f"STAC={ece['STAC-global'][0]:.3f} Or={ece['Oracle-TS'][0]:.3f}")

    print("=== feature-shift sweep ===");   do("feature", feat_levels)
    print("=== homophily-shift sweep ==="); do("homophily", homo_levels)
    print("=== mixed (heterogeneous) sweep ==="); do("mixed", mixed_levels)

    LEVELS = {"feature": feat_levels, "homophily": homo_levels, "mixed": mixed_levels}

    # ---------- save raw numbers ----------
    rows = []
    for kind in ["feature", "homophily", "mixed"]:
        for lv in LEVELS[kind]:
            for metric in ["ece", "mce", "brier", "nll", "auroc", "risk80"]:
                agg = aggregate(data[kind][lv], METHODS, metric)
                for mth in METHODS:
                    rows.append(dict(shift=kind, level=lv, metric=metric, method=mth,
                                     mean=agg[mth][0], std=agg[mth][1]))
    with open(f"{OUT}/stac_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["shift", "level", "metric", "method", "mean", "std"])
        w.writeheader(); w.writerows(rows)

    # ---------- Figure 1: ECE vs shift + accuracy-tracking (mechanism) ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = {"Uncal": "#888", "Source-TS": "#d62728", "STAC-global": "#1f77b4",
            "Oracle-TS": "#2ca02c"}
    for mth in ["Uncal", "Source-TS", "STAC-global", "Oracle-TS"]:
        mu = [aggregate(data["feature"][lv], METHODS, "ece")[mth][0] for lv in feat_levels]
        sd = [aggregate(data["feature"][lv], METHODS, "ece")[mth][1] for lv in feat_levels]
        ax[0].errorbar(feat_levels, mu, yerr=sd, marker="o", capsize=3,
                       label=mth, color=cols[mth])
    ax[0].set_xlabel("feature-shift intensity"); ax[0].set_ylabel("ECE (\u2193)")
    ax[0].set_title("Calibration under covariate shift"); ax[0].legend(); ax[0].grid(alpha=.3)

    true_acc = [np.mean([d["tgt_acc"] for d in diags["feature"][lv]]) for lv in feat_levels]
    a_hat = [np.mean([d["a_hat"] for d in diags["feature"][lv]]) for lv in feat_levels]
    src_a = [np.mean([d["src_val_acc"] for d in diags["feature"][lv]]) for lv in feat_levels]
    ax[1].plot(feat_levels, true_acc, "k-o", label="true target acc")
    ax[1].plot(feat_levels, a_hat, "b--s", label="STAC estimate (label-free)")
    ax[1].plot(feat_levels, src_a, "r:^", label="Source-TS assumption")
    ax[1].set_xlabel("feature-shift intensity"); ax[1].set_ylabel("accuracy")
    ax[1].set_title("Why it works: STAC tracks target accuracy")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_ece_vs_shift.png", dpi=140)

    # ---------- Figure 2: reliability diagrams ----------
    if rep_run is not None:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
        y = rep_run["y"]
        for j, name in enumerate(["Uncal", "Source-TS", "STAC-full"]):
            P = rep_run["P_reps"][name]
            conf = P.max(1); correct = (P.argmax(1) == y).astype(float)
            mids, accs, confs, ws = reliability(conf, correct, n_bins=12)
            e, _ = ece_mce(conf, correct)
            ax[j].plot([0, 1], [0, 1], "k--", alpha=.5)
            ax[j].bar(mids, accs, width=1 / 12 * .9, color="#1f77b4",
                      edgecolor="k", alpha=.8)
            ax[j].set_title(f"{name}  (ECE={e:.3f})")
            ax[j].set_xlabel("confidence"); ax[j].set_xlim(0, 1); ax[j].set_ylim(0, 1)
            if j == 0:
                ax[j].set_ylabel("accuracy")
        fig.suptitle("Reliability diagrams  (feature shift = 0.8)")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig2_reliability.png", dpi=140)

    # ---------- Figure 3: per-metric bars at representative shifts ----------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a, (kind, lv, ttl) in zip(
            ax, [("feature", 0.8, "Feature shift (0.8)"),
                 ("homophily", 0.25, "Homophily shift (0.25)")]):
        show = ["Uncal", "Source-TS", "STAC-global", "STAC-full", "Oracle-TS"]
        agg = aggregate(data[kind][lv], METHODS, "ece")
        mu = [agg[m][0] for m in show]; sd = [agg[m][1] for m in show]
        bar_c = ["#888", "#d62728", "#1f77b4", "#17becf", "#2ca02c"]
        a.bar(range(len(show)), mu, yerr=sd, capsize=3, color=bar_c)
        a.set_xticks(range(len(show))); a.set_xticklabels(show, rotation=20, ha="right")
        a.set_ylabel("ECE (\u2193)"); a.set_title(ttl); a.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_ece_bars.png", dpi=140)

    # ---------- Figure 4: error-detection AUROC (isolates PER-NODE benefit) ----------
    # A global temperature preserves cross-node confidence ranking -> AUROC == Uncal.
    # Only per-node temperature can raise AUROC, so this figure isolates the
    # contribution of the structure-aware per-node stage.
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for a, (kind, lv, ttl) in zip(ax, [
            ("feature", 1.1, "Feature shift (1.1)"),
            ("homophily", 0.25, "Homophily shift (0.25)"),
            ("mixed", 1.2, "Heterogeneous shift (1.2)")]):
        show = ["Source-TS", "STAC-global", "STAC-noStruct", "STAC-full"]
        agg = aggregate(data[kind][lv], METHODS, "auroc")
        mu = [agg[m][0] for m in show]; sd = [agg[m][1] for m in show]
        a.bar(range(len(show)), mu, yerr=sd, capsize=3,
              color=["#d62728", "#1f77b4", "#9467bd", "#17becf"])
        a.set_xticks(range(len(show))); a.set_xticklabels(show, rotation=20, ha="right")
        a.set_ylabel("error-detection AUROC (\u2191)"); a.set_title(ttl)
        a.set_ylim(0.5, max(0.75, max(mu) + 0.05)); a.grid(alpha=.3, axis="y")
    fig.suptitle("Only per-node recalibration improves error detection "
                 "(Source-TS = STAC-global by construction)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_auroc.png", dpi=140)

    # ---------- markdown summary table ----------
    def md_table(kind, levels, metric="ece", arrow="down"):
        head = "ECE" if metric == "ece" else metric.upper()
        lines = [f"### {kind}-shift {head} (mean\u00b1std over {len(SEEDS)} seeds)",
                 "| level | tgt_acc | Uncal | Source-TS | STAC-global | STAC-noStruct | STAC-full | Oracle-TS |",
                 "|--|--|--|--|--|--|--|--|"]
        for lv in levels:
            agg = aggregate(data[kind][lv], METHODS, metric)
            ta = np.mean([d["tgt_acc"] for d in diags[kind][lv]])
            def c(m): return f"{agg[m][0]:.3f}\u00b1{agg[m][1]:.3f}"
            lines.append(f"| {lv} | {ta:.2f} | {c('Uncal')} | {c('Source-TS')} | "
                         f"{c('STAC-global')} | {c('STAC-noStruct')} | "
                         f"**{c('STAC-full')}** | {c('Oracle-TS')} |")
        return "\n".join(lines)

    summary = (
        md_table("feature", feat_levels) + "\n\n" +
        md_table("homophily", homo_levels) + "\n\n" +
        md_table("mixed", mixed_levels) + "\n\n" +
        "### Error-detection AUROC \u2014 isolates the per-node contribution\n"
        "*(a single global temperature cannot change AUROC, so Source-TS = "
        "STAC-global; any gain is purely from structure-aware per-node scaling)*\n\n" +
        md_table("mixed", mixed_levels, metric="auroc") + "\n")
    with open(f"{OUT}/stac_summary.md", "w") as f:
        f.write("# STAC first results\n\n" + summary)
    print("\n" + summary)
    print("\nSaved: fig1_ece_vs_shift.png, fig2_reliability.png, fig3_ece_bars.png, "
          "fig4_auroc.png, stac_results.csv, stac_summary.md")
