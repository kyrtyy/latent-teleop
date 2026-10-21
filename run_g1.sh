#!/usr/bin/env bash
# Full G1 run on one GPU server. Usage: ./run_g1.sh <tag> [extra train.py args]
#   ./run_g1.sh main                          the G1 model
#   ./run_g1.sh no_rollout --rollout-weight 0 ablation: teacher forcing only
# Data is seeded and bit-identical across servers, so each server builds its own copy (skipped if present).
set -euo pipefail
cd "$(dirname "$0")"
TAG=$1; shift
PY="nice .venv/bin/python"
D=data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f $D/episodes/ep_01199.npz ] || $PY collect.py --out $D/episodes --per-profile 200   # 1200 episodes, ~1 min
[ -f $D/emb_p8.npy ] || $PY embed.py --data $D                                            # ~35 GB fp16
[ -f $D/probe_p8.pt ] || $PY train.py --data $D --probe-only --steps 20000 --batch 256 --lr 1e-3 --eval-every 5000 | tee $D/probe.log  # floor: look here ~40 min in
$PY train.py --data $D --out $D/wm_$TAG.pt "$@" | tee -a $D/train_$TAG.log                  # ~10 h on an RTX 3050
$PY eval_g1.py --data $D --ckpt $D/wm_$TAG.pt --probe-ckpt $D/probe_p8.pt --out results/$TAG
echo "G1 RUN DONE: results/$TAG"
