"""RQ4: a state-based diffusion policy trained on robot-side demonstrations (state, executed command), with the
segments a curation arm drops removed. Evaluated autonomously (no network) on paired held-out scenes.

Arms (segment = c4.SEG ticks):
  full                 every demonstration
  oracle               drop the true fault windows
  oracle_<kind>        drop only that fault kind's windows
  random_<arm>         drop as many random segments as <arm> drops (contiguous blocks)
  wm, rules            drop the segments flagged in <data>/flags.npz (c4.py flags)
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn

from c4 import KINDS, SEG
from sim import AGENT, ANGLE, BLOCK, make_env
from world_model import norm_xy

T_OBS, T_PRED, T_ACT, STEPS, DDIM = 2, 16, 8, 100, 10
AB = torch.cos((torch.arange(STEPS + 1) / STEPS + 0.008) / 1.008 * torch.pi / 2) ** 2
AB = (AB / AB[0])[1:].clamp(1e-4, 1)  # cosine schedule: alpha-bar at noise levels 1..STEPS


def obs_of(S):
    th = S[..., ANGLE:ANGLE + 1]
    return np.concatenate([norm_xy(S[..., AGENT]), norm_xy(S[..., BLOCK]), np.sin(th), np.cos(th)], -1).astype(np.float32)


def segments(L):
    return (len(L["exec"]) + SEG - 1) // SEG


def oracle_mask(L, kinds=KINDS):
    """Segment i is a fault segment if it overlaps the fault's effect window (same rule as c4.score)."""
    m = np.zeros(segments(L), bool)
    if str(L["fault"]) in kinds:
        s0 = np.arange(len(m)) * SEG
        m = (s0 < int(L["fault_t1"])) & (s0 + SEG > int(L["fault_t0"]))
    return m


def masks(data, arm, seed):
    files = sorted(glob.glob(f"{data}/episodes/ep_*.npz"))
    logs = [np.load(f) for f in files]
    base = arm.removeprefix("random_")
    if base == "full":
        m = [np.zeros(segments(L), bool) for L in logs]
    elif base == "oracle" or base.startswith("oracle_"):
        m = [oracle_mask(L, KINDS if base == "oracle" else [base[7:]]) for L in logs]
    else:
        flags = np.load(f"{data}/flags.npz", allow_pickle=True)[base]
        m = [np.asarray(f, bool)[:segments(L)] for f, L in zip(flags, logs)]
    if arm.startswith("random_"):  # same number of segments, chosen uniformly over all segments
        n, sizes = sum(x.sum() for x in m), [segments(L) for L in logs]
        pick = np.random.default_rng([seed, 99]).choice(sum(sizes), n, replace=False)
        flat = np.zeros(sum(sizes), bool)
        flat[pick] = True
        m = np.split(flat, np.cumsum(sizes)[:-1])
    return logs, m


def dataset(logs, drop):
    O, A = [], []
    for L, d in zip(logs, drop):
        S, E = L["state"], L["exec"]
        n = len(E)
        if n < T_PRED + T_OBS:
            continue
        bad = np.repeat(d, SEG)[:n]
        cb = np.concatenate([[0], np.cumsum(bad)])
        t = np.arange(T_OBS - 1, n - T_PRED + 1)
        t = t[cb[t + T_PRED] - cb[t - T_OBS + 1] == 0]  # no dropped tick in the observation or the action chunk
        o = obs_of(S)
        O.append(np.concatenate([o[t - T_OBS + 1 + i] for i in range(T_OBS)], -1))
        A.append(norm_xy(E[t[:, None] + np.arange(T_PRED)]).reshape(len(t), -1).astype(np.float32))
    return torch.from_numpy(np.concatenate(O)), torch.from_numpy(np.concatenate(A))


class Net(nn.Module):
    """Noise predictor: residual MLP over (noisy action chunk, observation, noise level)."""

    def __init__(self, d=512, depth=4):
        super().__init__()
        self.inp = nn.Linear(T_PRED * 2 + T_OBS * 6 + 64, d)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                    for _ in range(depth))
        self.out = nn.Linear(d, T_PRED * 2)
        self.register_buffer("freq", torch.exp(-np.log(1000) * torch.arange(32) / 32))

    def forward(self, x, o, k):
        e = k[:, None].float() * self.freq
        h = self.inp(torch.cat([x, o, e.sin(), e.cos()], -1))
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


def train(O, A, steps, seed, batch=256, lr=3e-4):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = Net()
    ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    for step in range(steps):
        i = torch.randint(len(O), (batch,), generator=g)
        k = torch.randint(STEPS, (batch,), generator=g)
        eps = torch.randn(batch, A.shape[1], generator=g)
        ab = AB[k][:, None]
        loss = ((net(ab.sqrt() * A[i] + (1 - ab).sqrt() * eps, O[i], k) - eps) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        ema.update_parameters(net)
        if step % 5000 == 0:
            print(f"  step {step} loss {loss.item():.4f}", flush=True)
    return ema.module.eval()


@torch.no_grad()
def sample(net, o, g):
    x = torch.randn(len(o), T_PRED * 2, generator=g)
    ks = np.linspace(STEPS - 1, 0, DDIM).round().astype(int)
    for i, k in enumerate(ks):
        ab, ab_next = AB[k], AB[ks[i + 1]] if i + 1 < len(ks) else torch.tensor(1.0)
        eps = net(x, o, torch.full((len(o),), int(k)))
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-2.5, 2.5)
        x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps  # deterministic DDIM
    return x.view(-1, T_PRED, 2) * 128 + 256


@torch.no_grad()
def evaluate(net, seeds, max_ticks=1000):
    """All episodes in lockstep. -> coverage curves (n, max_ticks + 1), held after success; success flags."""
    from sim import full_state
    envs = [make_env(s) for s in seeds]
    g = torch.Generator().manual_seed(12345)
    prev = cur = np.stack([full_state(e) for e in envs])
    cov = np.zeros((len(envs), max_ticks + 1))
    cov[:, 0] = [e._get_coverage() for e in envs]
    done = np.zeros(len(envs), bool)
    for t in range(max_ticks):
        if t % T_ACT == 0:
            plan = sample(net, torch.from_numpy(np.concatenate([obs_of(prev), obs_of(cur)], -1)), g).numpy()
        prev, nxt = cur, cur.copy()
        for i, e in enumerate(envs):
            if done[i]:
                cov[i, t + 1] = cov[i, t]
                continue
            _, _, done[i], _, _ = e.step(plan[i, t % T_ACT])
            nxt[i] = full_state(e)
            cov[i, t + 1] = e._get_coverage()
        cur = nxt
    return cov, done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_rq4")
    ap.add_argument("--arm", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--eval-seed", type=int, default=31)
    ap.add_argument("--out", default="results/rq4")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    f = f"{args.out}/{args.arm}_s{args.seed}.npz"
    if os.path.exists(f):
        raise SystemExit(f"{f} exists")
    t0 = time.time()
    logs, drop = masks(args.data, args.arm, args.seed)
    O, A = dataset(logs, drop)
    n_drop, n_seg = sum(d.sum() for d in drop), sum(len(d) for d in drop)
    print(f"{args.arm} s{args.seed}: dropped {n_drop}/{n_seg} segments, {len(O)} training windows", flush=True)
    net = train(O, A, args.steps, args.seed)
    torch.save(net.state_dict(), f[:-4] + ".pt")
    seeds = [args.eval_seed * 1_000_003 + i for i in range(args.n_eval)]
    cov, success = evaluate(net, seeds)
    np.savez(f, coverage=cov, success=success, seeds=seeds, dropped=n_drop, segments=n_seg, windows=len(O))
    print(f"{args.arm} s{args.seed}: coverage AUC {cov.mean():.3f} success {success.mean():.2f} ({time.time() - t0:.0f}s)")
