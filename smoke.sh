#!/usr/bin/env bash
# End-to-end smoke test on a tiny dataset: self-checks -> collect -> embed -> probe -> train -> G1 eval.
# Fails loudly on any error. A few minutes on one RTX 3050. Numbers it prints are meaningless (tiny data).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
D=data_smoke
rm -rf $D results_smoke
$PY sim.py
$PY teleop.py
$PY collect.py --out $D/episodes --per-profile 5 --max-ticks 300
$PY embed.py --data $D --pool 8
$PY train.py --data $D --pool 8 --probe-only --steps 200 --batch 256 --lr 1e-3 --eval-every 100
$PY train.py --data $D --pool 8 --steps 100 --eval-every 50
$PY eval_g1.py --data $D --pool 8 --out results_smoke
test -s results_smoke/g1.png
echo "SMOKE OK"
