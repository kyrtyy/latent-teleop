"""Perception-floor diagnostics: which readout of a frame places the agent and block to within a few px?
  --input tokens --probe mlp      flatten-MLP on DINOv2 tokens (the probe train.py uses)
  --input tokens --probe spatial  per-token scores + offsets, softmax-weighted token centres (translation-equivariant)
  --input pixels                  small CNN on 56x56 renders: a sanity floor (a disk on white should be ~1-2 px)
Reports train and val error with the same metric, so under- and over-fitting are visible."""
import argparse
import json
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim import make_env, render
from world_model import Probe, episode_split, decode, norm, probe_targets

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_diag")
ap.add_argument("--input", choices=["tokens", "pixels"], default="tokens")
ap.add_argument("--probe", choices=["mlp", "spatial"], default="spatial")
ap.add_argument("--pool", type=int, default=4)
ap.add_argument("--episodes", type=int, default=10**9, help="cap on training episodes")
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--lr", type=float, default=1e-3)
args = ap.parse_args()
torch.manual_seed(0)
rng = np.random.default_rng(0)
dev = "cuda"

meta = np.load(f"{args.data}/meta.npz")
off = meta["offsets"]
split = episode_split(meta["profiles"])
pick = lambda eps: np.concatenate([np.arange(off[e], off[e + 1]) for e in eps])
train_f = pick(np.flatnonzero(split >= 2)[:args.episodes])
val_all = pick(np.flatnonzero(split == 1))
val_f = rng.choice(val_all, min(4096, len(val_all)), replace=False)
tgt = torch.from_numpy(probe_targets(meta["states"], meta["contact"]))[:, :6]


def _render56(states):
    env = make_env()
    return np.stack([cv2.resize(render(env, s), (56, 56), interpolation=cv2.INTER_AREA) for s in states])


if args.input == "pixels":
    used = np.concatenate([train_f, val_f])
    with Pool(16) as p:
        imgs = np.concatenate(p.map(_render56, np.array_split(meta["states"][used], 64)))
    X = torch.zeros((len(meta["states"]), 3, 56, 56), dtype=torch.uint8)
    X[used] = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    get = lambda i: X[i].to(dev).float() / 255
    net = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                        nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(), nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
                        nn.Flatten(), nn.Linear(128 * 7 * 7, 256), nn.GELU(), nn.Linear(256, 6))
else:
    emb = torch.from_numpy(np.load(f"{args.data}/emb_p{args.pool}.npy"))
    get = lambda i: norm(emb[torch.from_numpy(i)].to(dev))
    n_tok = emb.shape[1]

    if args.probe == "mlp":
        net = nn.Sequential(nn.Flatten(-2), nn.Linear(n_tok * 384, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 6))
    else:
        class Pose(Probe):  # same probe as training, minus the contact output
            def forward(self, z):
                return super().forward(z)[..., :6]

        net = Pose(n_tok)

net = net.to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)


@torch.no_grad()
def err(frames):
    net.eval()
    out = torch.cat([net(get(frames[i:i + 512])) for i in range(0, len(frames), 512)])
    net.train()
    pad = lambda x: torch.cat([x, torch.zeros(len(x), 1, device=dev)], 1)
    (ag, bl, th, _), (ag_t, bl_t, th_t, _) = decode(pad(out)), decode(pad(tgt[frames].to(dev)))
    return {"agent_px": round((ag - ag_t).norm(dim=-1).mean().item(), 2), "block_px": round((bl - bl_t).norm(dim=-1).mean().item(), 2),
            "angle_deg": round(torch.rad2deg(torch.remainder(th - th_t + torch.pi, 2 * torch.pi) - torch.pi).abs().mean().item(), 2)}


train_eval = rng.choice(train_f, 4096, replace=False)
for step in range(1, args.steps + 1):
    i = rng.choice(train_f, 256)
    loss = F.mse_loss(net(get(i)), tgt[i].to(dev))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
name = f"{args.input}" + ("" if args.input == "pixels" else f"-{args.probe}-p{args.pool}")
print(json.dumps({"probe": name, "train_episodes": int(min(args.episodes, (split >= 2).sum())), "train_frames": len(train_f),
                  "train": err(train_eval), "val": err(val_f)}), flush=True)
