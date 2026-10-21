"""Is the learned model's G1 win real dynamics, or does it read the future from somewhere it shouldn't?
Re-runs the G1 rollout (block-moving windows) with the future commands corrupted:
  true      the executed commands, as in eval_g1.py
  hold-cmd  every future command = the last one before the anchor (no information about what comes next)
  shuffled  future commands taken from another anchor
If the gain comes from physics driven by the commands, both corruptions should drop it to about hold-last-frame.
Per delay profile: with >=300 ms delay the operator issued the window's commands from frames before the anchor,
so commands cannot carry operator reactions to the future block. A similar gain there rules out that leak."""
import argparse

import numpy as np
import torch

from sim import ANGLE, BLOCK, wrap
from world_model import Predictor, Probe, amp, decode, episode_split, norm, norm_xy, rollout

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--ckpt", default="data/wm_main.pt")
ap.add_argument("--stride", type=int, default=3)
args = ap.parse_args()
dev = "cuda"
ck = torch.load(args.ckpt, weights_only=False)
H, K, n_tok = ck["args"]["H"], ck["args"]["K"], ck["n_tok"]
model, probe = Predictor(n_tok, max_t=H + K).to(dev).eval(), Probe(n_tok).to(dev).eval()
model.load_state_dict(ck["model"])
probe.load_state_dict(torch.load(f"{args.data}/probe_p8.pt", weights_only=False)["probe"])

meta = np.load(f"{args.data}/meta.npz")
emb = torch.from_numpy(np.load(f"{args.data}/emb_p8.npy", mmap_mode="r"))
S, act, off, prof = meta["states"], meta["action"], meta["offsets"], meta["profiles"]
test = np.flatnonzero(episode_split(prof) == 0)
anchors = np.concatenate([np.arange(off[e] + H - 1, off[e + 1] - K, args.stride) for e in test])
ep_prof = prof[np.searchsorted(off, anchors, "right") - 1]
fut = anchors[:, None] + np.arange(1, K + 1)
truth = S[fut]
moving = (np.linalg.norm(truth[..., BLOCK] - S[anchors, None, BLOCK], axis=-1) >= 5) | \
         (np.degrees(np.abs(wrap(truth[..., ANGLE] - S[anchors, None, ANGLE]))) >= 3)
perm = np.random.default_rng(0).permutation(len(anchors))


@torch.no_grad()
def run(mode):
    bl, th = [], []
    for i in range(0, len(anchors), 128):
        a = anchors[i:i + 128]
        z = norm(emb[torch.from_numpy(a[:, None] + np.arange(-H + 1, 1))].to(dev))  # context only, no future frames
        u = act[a[:, None] + np.arange(-H + 1, K)].copy()  # u[:, H-1:] = commands executed at ticks >= anchor
        if mode == "hold-cmd":
            u[:, H - 1:] = u[:, H - 2:H - 1]
        elif mode == "shuffled":
            b = anchors[perm[i:i + 128]]
            u[:, H - 1:] = act[b[:, None] + np.arange(K)]
        with amp():
            pred = rollout(model, z, torch.from_numpy(norm_xy(u).astype(np.float32)).to(dev), K).float()
        _, b_, t_, _ = decode(probe(pred))
        bl.append(b_.cpu().numpy()), th.append(t_.cpu().numpy())
    bl, th = np.concatenate(bl), np.concatenate(th)
    return np.linalg.norm(bl - truth[..., BLOCK], axis=-1), np.degrees(np.abs(wrap(th - truth[..., ANGLE])))


hold = (np.linalg.norm(S[anchors, None, BLOCK] - truth[..., BLOCK], axis=-1),
        np.degrees(np.abs(wrap(S[anchors, None, ANGLE] - truth[..., ANGLE]))))
res = {"hold-last-frame (true)": hold} | {m: run(m) for m in ["true", "hold-cmd", "shuffled"]}
groups = {"all profiles": np.ones(len(anchors), bool)} | {p: ep_prof == p for p in np.unique(prof)}
print(f"block-moving windows, block px / angle deg at 300 ms and 600 ms")
for g, sel in groups.items():
    print(f"\n{g}")
    for name, (b, t) in res.items():
        cells = [f"{b[sel & moving[:, k], k].mean():5.1f}px {t[sel & moving[:, k], k].mean():5.1f}deg" for k in (9, 19)]
        print(f"  {name:24s} " + "   ".join(cells))
