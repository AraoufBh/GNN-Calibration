#!/usr/bin/env bash
# Reproduce all figures. Run from the repository root:  bash scripts/run_all.sh
set -e
mkdir -p figures
echo "== theory (Fig 5) =="                 ; python code/theory.py
echo "== extensions (Fig 6) =="             ; python code/theory_ext.py
echo "== ECE bound (Fig 7) =="              ; python code/bound_check.py
echo "== method, synthetic (Figs 1-4) =="   ; python code/stac.py
echo "== optimal-temperature (Fig 10) =="   ; python code/temp_validation.py
echo "== download real graphs =="           ; bash data/download_data.sh
echo "== real: minesweeper (Fig 8) =="      ; python code/real_data.py
echo "== real: 5 graphs =="                 ; python code/real_data_multi.py
echo "== real figure (Fig 9) =="            ; python code/build_fig_real.py
echo "== covariate sweep =="                ; python code/real_shift_sweep.py
echo "== structural shift =="               ; python code/real_structural_shift.py
echo "== real shift figures (Figs 11-12) ==" ; python code/build_new_figs.py
echo "All figures written to ./figures/"
