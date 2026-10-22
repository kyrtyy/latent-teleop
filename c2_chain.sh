#!/usr/bin/env bash
# After the world model: 4x-data probe (floor.sh), then the learned-ghost C2 cells.
cd "$(dirname "$0")"
until grep -q "G1 RUN DONE" run_main.log 2>/dev/null; do sleep 300; done
./floor.sh > floor.log 2>&1
nice .venv/bin/python c2.py --kinds learned > c2_learned.log 2>&1
echo "C2 CHAIN DONE" >> c2_learned.log
