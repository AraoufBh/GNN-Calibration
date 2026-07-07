#!/usr/bin/env bash
# Downloads the 5 real graphs (Platonov et al., ICLR 2023) into this data/ folder.
set -e
cd "$(dirname "$0")"
BASE="https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data"
for f in roman_empire amazon_ratings tolokers minesweeper questions; do
  echo "downloading ${f}.npz ..."
  curl -L -o "${f}.npz" "${BASE}/${f}.npz"
done
echo "done. (synthetic experiments need no download.)"
