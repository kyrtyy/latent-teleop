"""Train the probe alone (--probe-only: the perception floor, used to pick the pooling) or predictor + probe (C1)."""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA init: 6 GB cards fragment
import json
import time

import numpy as np
import torch

from world_model import amp, episode_split, Predictor, Probe, decode, norm, norm_xy, probe_loss, probe_targets, rollout

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--pool", type=int, default=8)
ap.add_argument("--probe-only", action="store_true")
ap.add_argument("--steps", type=int, default=40000)
ap.add_argument("--batch", type=int, default=16)  # 8x8 pooling: 16 peaks at 5.0 GB, 0.87 s/step on an RTX 3050
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--H", type=int, default=4, help="context frames")
ap.add_argument("--K", type=int, default=20, help="rollout ticks (20 x 30 ms = 600 ms)")
ap.add_argument("--rollout-weight", type=float, default=1.0, help="0 = teacher forcing only (ablation)")
ap.add_argument("--eval-every", type=int, default=1000)
ap.add_argument("--out", default=None)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
dev = "cuda"

meta = np.load(f"{args.data}/meta.npz")
emb = torch.from_numpy(np.load(f"{args.data}/emb_p{args.pool}.npy", mmap_mode="r"))  # fp16 memmap (~35 GB at 8x8)
targets = torch.from_numpy(probe_targets(meta["states"], meta["contact"]))
actions = torch.from_numpy(norm_xy(meta["action"]).astype(np.float32))
offsets, n_tok = meta["offsets"], emb.shape[1]
split = episode_split(meta["profiles"])


def frames(part):
    return np.concatenate([np.arange(offsets[e], offsets[e + 1]) for e in np.flatnonzero(part)])


def windows(part, W):
    return np.concatenate([np.arange(offsets[e], offsets[e + 1] - W + 1) for e in np.flatnonzero(part)])


probe = Probe(n_tok).to(dev)
params = list(probe.parameters())
if not args.probe_only:
    W = args.H + args.K
    model = Predictor(n_tok, max_t=W).to(dev)
    params += list(model.parameters())
    train_w, val_w = windows(split >= 2, W), windows(split == 1, W)
    val_w = rng.choice(val_w, min(1024, len(val_w)), replace=False)
    print(f"{len(train_w)} train windows, predictor {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
train_f, val_f = frames(split >= 2), frames(split == 1)
val_f = rng.choice(val_f, min(4096, len(val_f)), replace=False)
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)


def batch(starts, W):
    idx = torch.from_numpy(starts[:, None] + np.arange(W))
    return norm(emb[idx].to(dev)), actions[idx].to(dev), targets[idx].to(dev)


@torch.no_grad()
def evaluate():
    probe.eval()
    z, _, tgt = batch(val_f, 1)
    (ag, bl, th, c), (ag_t, bl_t, th_t, _) = decode(probe(z)), decode(tgt)
    res = {"probe_agent_px": (ag - ag_t).norm(dim=-1).mean().item(),
           "probe_block_px": (bl - bl_t).norm(dim=-1).mean().item(),
           "probe_angle_deg": torch.rad2deg(torch.remainder(th - th_t + torch.pi, 2 * torch.pi) - torch.pi).abs().mean().item(),
           "probe_contact_acc": ((c > 0.5) == (tgt[..., 6] > 0.5)).float().mean().item()}
    if not args.probe_only:
        model.eval()
        acc = []
        for i in range(0, len(val_w), 64):  # small chunks: training memory is still resident
            z, a, tgt = batch(val_w[i:i + 64], W)
            with amp():
                pred = rollout(model, z[:, :args.H], a, args.K).float()
            bl = decode(probe(pred[:, -1]))[1]
            acc.append(torch.stack([(pred[:, -1] - z[:, -1]).abs().mean((-1, -2)), (pred[:, 0] - z[:, args.H]).abs().mean((-1, -2)),
                                    (bl - decode(tgt[:, -1])[1]).norm(dim=-1)], 1))
        acc = torch.cat(acc).mean(0).tolist()
        res |= dict(zip(["rollout_l1_final", "rollout_l1_first", "rollout_block_px_final"], acc))
        model.train()
    probe.train()
    return res


out = args.out or f"{args.data}/{'probe' if args.probe_only else 'wm'}_p{args.pool}.pt"
ckpt, start = out + ".resume", 1  # resume after a kill: the step, optimiser and data RNG continue where they stopped
if os.path.exists(ckpt):
    c = torch.load(ckpt, weights_only=False)
    probe.load_state_dict(c["probe"]), opt.load_state_dict(c["opt"]), sched.load_state_dict(c["sched"])
    if not args.probe_only:
        model.load_state_dict(c["model"])
    rng.bit_generator.state, start = c["rng"], c["step"] + 1
    print("resumed from step", c["step"], flush=True)
t0 = time.time()
for step in range(start, args.steps + 1):
    if args.probe_only:
        z, _, tgt = batch(rng.choice(train_f, args.batch), 1)  # run_g1.sh: --batch 256 --lr 1e-3, as probe_check.py
        loss = probe_loss(probe(z), tgt)
    else:
        z, a, tgt = batch(rng.choice(train_w, args.batch), W)
        with amp():
            teacher = (model(z, a)[:, :-1].float() - z[:, 1:]).abs().mean()  # every next-frame, from true history
            roll = (rollout(model, z[:, :args.H], a, args.K).float() - z[:, args.H:]).abs().mean()  # own predictions fed back
        p = probe_loss(probe(z), tgt)  # probe learns from true frames only; applied to predictions at eval
        loss = teacher + args.rollout_weight * roll + p
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    if step % args.eval_every == 0 or step == args.steps:
        print(json.dumps({"step": step, "loss": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 1),
                          **{k: round(v, 4) for k, v in evaluate().items()}}), flush=True)
        torch.save({"step": step, "probe": probe.state_dict(), "model": None if args.probe_only else model.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.bit_generator.state}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

torch.save({"args": vars(args), "probe": probe.state_dict(), "model": None if args.probe_only else model.state_dict(),
            "n_tok": n_tok, "final": evaluate()}, out)
print("saved", out)
