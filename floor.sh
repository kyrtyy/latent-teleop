#!/usr/bin/env bash
# Perception floor on 4x data: same episodes 0-1199 (seeded) + 3600 more. Retrains only the probe; the predictor is reused.
set -euo pipefail
cd "$(dirname "$0")"
PY="nice .venv/bin/python"; D=data_big
[ -f $D/episodes/ep_04799.npz ] || $PY collect.py --out $D/episodes --per-profile 800
[ -f $D/emb_p8.npy ] || $PY embed.py --data $D
$PY train.py --data $D --probe-only --steps 150000 --batch 256 --lr 1e-3 --eval-every 10000 --out $D/probe150k_p8.pt | tee -a $D/probe150k.log
$PY eval_g1.py --data data --ckpt data/wm_main.pt --probe-ckpt $D/probe150k_p8.pt --out results/main_probe150k
echo "FLOOR RUN DONE"
