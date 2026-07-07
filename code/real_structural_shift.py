"""
Experiment C: structural shift (edge rewiring) on real graphs.
We rewire a fraction f of edges (random endpoint replacement), which drives edge homophily
toward chance and thus changes the calibration slope kappa (Thm 1 / Prop 3). Prediction:
structural shift miscalibrates the frozen model, and a single oracle temperature restores
calibration -- a shift type distinct from covariate noise.
"""
import json, os
import numpy as np
import scipy.sparse as sp
from real_shift_sweep import (load, build_S, sign_features, softmax, to_logits,
                              opt_T, ece)
from sklearn.linear_model import LogisticRegression

DATASETS = ["roman_empire", "amazon_ratings", "tolokers", "minesweeper", "questions"]
FRACS = [0.0, 0.25, 0.5, 0.75]
JSON = "data/results_structural.json"


def rewire(edges, N, frac, rng):
    E = edges.copy()
    m = rng.random(len(E)) < frac
    E[m, 1] = rng.integers(0, N, m.sum())          # replace destination endpoint
    E = E[E[:, 0] != E[:, 1]]                       # drop self-loops created
    return E


def process(name, splits=2):
    X, y, edges, trm, vam, tem = load(name)
    N = len(y); Sclean = build_S(edges, N); Fclean = sign_features(Sclean, X)
    res = {str(f): {m: [] for m in ["unc", "src", "orc", "hom"]} for f in FRACS}
    for sp_i in range(splits):
        tr, va, te = trm[sp_i], vam[sp_i], tem[sp_i]
        rng = np.random.default_rng(sp_i)
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200).fit(Fclean[tr], y[tr])
        Tsrc = opt_T(to_logits(clf, Fclean)[va], y[va])
        for f in FRACS:
            E = rewire(edges, N, f, rng)
            h = float((y[E[:, 0]] == y[E[:, 1]]).mean())
            S = build_S(E, N); L = to_logits(clf, sign_features(S, X))[te]
            res[str(f)]["unc"].append(ece(L, y[te], 1.0))
            res[str(f)]["src"].append(ece(L, y[te], Tsrc))
            res[str(f)]["orc"].append(ece(L, y[te], opt_T(L, y[te])))
            res[str(f)]["hom"].append(h)
    agg = {f: {m: float(np.mean(v)) for m, v in d.items()} for f, d in res.items()}
    return dict(name=name, fracs=FRACS, curve=agg)


if __name__ == "__main__":
    out = json.load(open(JSON)) if os.path.exists(JSON) else {}
    for name in DATASETS:
        if name in out:
            continue
        r = process(name); out[name] = r
        json.dump(out, open(JSON, "w"), indent=1)
        row = " ".join(f"f={f}:h={r['curve'][str(f)]['hom']:.2f},unc={r['curve'][str(f)]['unc']:.3f},"
                       f"orc={r['curve'][str(f)]['orc']:.3f}" for f in FRACS)
        print(f"{name:15s} {row}", flush=True)
    print("done")
