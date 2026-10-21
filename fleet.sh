#!/usr/bin/env bash
# Push code to servers (last octet of 10.81.30.x), build the venv there, run the self-checks. Usage: ./fleet.sh 91 70 55
set -uo pipefail
cd "$(dirname "$0")"
for h in "$@"; do
  (
    host=yashdewangan@10.81.30.$h
    rsync -az --exclude .git --exclude prototype --exclude 'data*' --exclude 'results*' ./ "$host:~/aktoria_teleop/" &&
    ssh -o BatchMode=yes "$host" 'set -e; cd ~/aktoria_teleop
      [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
      [ -d .venv ] || ~/.local/bin/uv venv -q -p 3.12 .venv
      ~/.local/bin/uv pip install -q -p .venv --index-strategy unsafe-best-match -r requirements.txt
      for f in sim teleop world_model; do .venv/bin/python $f.py 2>&1 | grep -v -i warn | tail -1; done
      .venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0), \"bf16\" if torch.cuda.is_bf16_supported(including_emulation=False) else \"fp32\")"' 2>&1 | sed "s/^/.$h  /"
  ) &
done
wait
