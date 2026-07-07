# When Does Distribution Shift Break GNN Calibration?
### A Closed-Form Analysis and a Label-Free Remedy

Code, data and paper for a study of **graph neural network (GNN) calibration under
distribution shift**. We prove a closed-form *calibration slope* $\kappa$ that determines
how shift makes a GNN over- or under-confident, and that the optimal fix is a single
temperature $T^\star = 1/\kappa$. Every theoretical claim is validated numerically, and we
report an honest negative result about the practical label-free method.

**One-line summary.** For a linear GNN on a contextual stochastic block model,
confidence $=\sigma(t)$ but accuracy $=\sigma(\kappa t)$; the model is over-confident iff
$\kappa<1$, under-confident iff $\kappa>1$, and $T^\star=1/\kappa$ removes the gap. A
single global temperature is therefore optimal (per-node recalibration is provably
unnecessary), which we confirm on synthetic and 5 real graphs.

Everything runs **on CPU** with only NumPy / SciPy / scikit-learn / Matplotlib.

---

## Repository layout
```
.
├── paper/
│   ├── paper.tex          # two-column paper (compiles with latexmk)
│   ├── references.bib     # references
│   ├── figures/           # figures used by the paper
│   ├── supplementary_proofs.md  # full proofs of all propositions
│   └── paper.pdf          # pre-compiled PDF
├── code/
│   ├── theory.py          # Thm 1 validation                       -> figures/fig5_*
│   ├── theory_ext.py      # self-loop GCN + K-class extensions      -> figures/fig6_*
│   ├── bound_check.py     # ECE bound                               -> figures/fig7_*
│   ├── stac.py            # method under synthetic shift + ablation -> figures/fig1-4_*
│   ├── temp_validation.py # predicted T*=1/kappa vs oracle T        -> figures/fig10_*
│   ├── real_data.py       # minesweeper covariate-shift sweep       -> figures/fig8_*
│   ├── real_data_multi.py # 5 real graphs                           -> data/results_real.json
│   ├── build_fig_real.py  # real-data figure                        -> figures/fig9_*
│   ├── real_shift_sweep.py    # covariate-shift sweep (5 graphs)    -> data/results_shift_sweep.json
│   ├── real_structural_shift.py # edge-rewiring shift (5 graphs)    -> data/results_structural.json
│   └── build_new_figs.py  # covariate + structural figures          -> figures/fig11_*, fig12_*
├── data/
│   └── download_data.sh   # fetches the 5 real graphs into data/
├── scripts/
│   └── run_all.sh         # reproduce every figure
├── requirements.txt
└── LICENSE
```

## Install
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce
```bash
# from the repository root
bash scripts/run_all.sh          # all figures -> ./figures/
```
or run pieces individually, e.g.
```bash
python code/theory.py            # Fig 5  (kappa: theory vs simulation)
python code/theory_ext.py        # Fig 6  (self-loop GCN, c_K(h))
python code/bound_check.py       # Fig 7  (ECE bound)
python code/stac.py              # Figs 1-4 (synthetic method + per-node study)
bash   data/download_data.sh     # real graphs (~94 MB total)
python code/real_data_multi.py   # 5 real graphs
python code/build_fig_real.py    # Fig 9
```
Synthetic experiments need **no** download. Runtimes are seconds to a couple of minutes
each on a single CPU core.

## Figure → script map
| Figure | Script | What it shows |
|---|---|---|
| 5 | `theory.py` | closed-form $\kappa$ matches $1/T_{\text{oracle}}$ across homophily |
| 6 | `theory_ext.py` | self-loop GCN slope $\kappa_{SL}$; $K$-class coefficient $c_K(h)$ |
| 7 | `bound_check.py` | $\mathrm{ECE}\le\frac14|\kappa-1|\,\mathbb{E}|\delta|$ |
| 10 | `temp_validation.py` | predicted $T^\star=1/\kappa$ vs oracle temperature ($r=0.99$) |
| 1–4 | `stac.py` | method vs shift; reliability; per-node ablation (AUROC) |
| 8 | `real_data.py` | minesweeper: oracle temperature recovers calibration |
| 9 | `real_data_multi.py` + `build_fig_real.py` | 5 real graphs across homophily |
| 11 | `real_shift_sweep.py` + `build_new_figs.py` | covariate-shift sweep; over-confidence grows (Prop 4) |
| 12 | `real_structural_shift.py` + `build_new_figs.py` | edge-rewiring (structural) shift |

## Key results
- **Theory is exact in simulation.** The closed-form slope, its self-loop-GCN and
  $K$-class extensions, and the ECE bound all match experiments (Figs 5–7). The closed form
  predicts the *optimal* temperature $T^\star=1/\kappa$ with Pearson $r=0.99$ (Fig 10).
- **Two shift types on real graphs.** Covariate shift makes the model progressively
  over-confident (Prop 4), verified as a monotone confidence$-$accuracy gap across all five
  graphs (Fig 11); structural edge-rewiring also miscalibrates and is recovered by a single
  oracle temperature (Fig 12).
- **A single global temperature suffices.** Validated on synthetic shift and on **five
  real graphs** (homophily 0.05–0.84, $K$ up to 18): one oracle temperature brings ECE to
  $\le 0.024$ everywhere. Per-node recalibration gives no benefit — as the theory predicts.
- **Honest negative result.** The *label-free* method halves the gap on synthetic shifts
  but is unreliable on real graphs, because label-free **accuracy estimation** under shift
  (ATC / disagreement) overshoots. We isolate this as the key open problem.

## Paper
Compile with `latexmk` (handles BibTeX automatically):
```bash
cd paper && latexmk -pdf paper.tex
```
`paper.tex` uses a standard `article` two-column class so it builds anywhere. **For
Neural Networks / Elsevier submission**, swap the first line to
`\documentclass[5p]{elsarticle}` and move the title/abstract into `\begin{frontmatter}`
(the content, math, figures and references are unchanged). The three method/illustration
figures (Figs 1, 2, and the homophily schematic) are drawn in **TikZ/PGFPlots** inside
`paper.tex`.

## Data
The five real graphs are from **Platonov et al., "A Critical Look at the Evaluation of
GNNs under Heterophily: Are We Really Making Progress?", ICLR 2023**, redistributed by the
authors at <https://github.com/yandex-research/heterophilous-graphs>. `download_data.sh`
fetches them; please cite the original work if you use them.

## Citation
```bibtex
@article{calibrationslope2025,
  title  = {When Does Distribution Shift Break Graph Neural Network Calibration?
            A Closed-Form Analysis and a Label-Free Remedy},
  author = {Anonymous},
  year   = {2025}
}
```

## License
MIT (see `LICENSE`). Dataset licenses are those of the original authors.
