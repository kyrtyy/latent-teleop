"""C4 self-auditing: inject labelled faults into teleoperated demonstrations, then score every segment.

Faults (one per faulty episode, window [t0, t1) in ticks; `effect` extends it by how long the fault shows):
  hesitate  the operator freezes the cursor for 20-40 ticks
  wrong     the operator pushes toward a random point for 15-30 ticks (a mistaken push)
  spike     both links get +700 ms on every packet sent during 25 ticks (a network spike)
  slip      the block jumps 15-30 px and 0.15-0.3 rad away from the agent (an unexpected contact / slip)
Natural hesitations are switched off (op hesitate = 0), so clean episodes are clean.

Segment scores (SEG ticks each), from frozen DINOv2 tokens and the C1 world model:
  dyn    k-step latent prediction error from the true context under the executed commands (physics surprise)
  stale  latent distance between the frame the operator acted on and the scene when its command ran
  blind  world-model error predicting that scene from the operator's frame under the executed commands
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from sim import AGENT, ANGLE, BLOCK, ScriptedOperator, full_state, make_env, set_state
from teleop import PROFILES, Delay, Link, run_episode

KINDS = ["hesitate", "wrong", "spike", "slip"]
SEG = 15


class FaultOperator:
    def __init__(self, op, kind, t0, t1, rng):
        self.op, self.kind, self.t0, self.t1 = op, kind, t0, t1
        self.goal = rng.uniform(40, 472, 2)
        self.params = op.params

    def act(self, s, t):
        if not self.t0 <= t < self.t1 or self.kind not in ("hesitate", "wrong"):
            return self.op.act(s, t)
        if self.kind == "wrong":
            step = self.goal - self.op.cursor
            self.op.cursor = self.op.cursor + step * min(1.0, self.params["speed"] / max(np.linalg.norm(step), 1e-6))
        return self.op.cursor.copy()


class SpikeDelay:
    """One call per packet and one packet per tick, so the call count is the send tick."""

    def __init__(self, delay, t0, t1, extra=700.0):
        self.delay, self.t0, self.t1, self.extra, self.i = delay, t0, t1, extra, 0

    def __call__(self):
        d = self.delay() + (self.extra if self.t0 <= self.i < self.t1 else 0.0)
        self.i += 1
        return d


class SlipEnv:
    def __init__(self, env, t0, rng):
        self.env, self.t0, self.t, self.rng = env, t0, 0, rng

    def __getattr__(self, k):
        return getattr(self.env, k)

    def step(self, a):
        if self.t == self.t0:
            s = full_state(self.env)
            away = s[BLOCK] - s[AGENT]
            s[BLOCK] += away / max(np.linalg.norm(away), 1e-6) * self.rng.uniform(15, 30)
            s[ANGLE] += self.rng.choice([-1, 1]) * self.rng.uniform(0.15, 0.3)
            s[BLOCK] = np.clip(s[BLOCK], 60, 452)
            set_state(self.env, s)
        self.t += 1
        return self.env.step(a)


def episode(seed, profile, kind, max_ticks=600):
    """kind None = clean. Returns the teleop log plus fault labels."""
    rng = np.random.default_rng(seed)
    env = make_env(seed)
    op = ScriptedOperator(rng, full_state(env)[AGENT])
    op.params["hesitate"] = 0.0
    up, down = Delay(profile, rng), Delay(profile, rng)
    frng = np.random.default_rng([seed, 1])  # fault draws never shift the operator or delay streams
    t0 = int(frng.integers(60, 400))
    L = {"hesitate": int(frng.integers(20, 41)), "wrong": int(frng.integers(15, 31)), "spike": 25, "slip": 1}.get(kind, 0)
    if kind in ("hesitate", "wrong"):
        op = FaultOperator(op, kind, t0, t0 + L, frng)
    elif kind == "spike":
        up, down = SpikeDelay(up, t0, t0 + L), SpikeDelay(down, t0, t0 + L)
    elif kind == "slip":
        env = SlipEnv(env, t0, frng)
    log = run_episode(env, op, Link(up), Link(down), max_ticks)
    effect = {"hesitate": 10, "wrong": 15, "spike": 35, "slip": 15}.get(kind, 0)
    return log | dict(fault=kind or "clean", fault_t0=t0 if kind else -1, fault_t1=t0 + L + effect if kind else -1,
                      profile=profile, seed=seed)


def _one(job):
    out, idx, seed, profile, kind = job
    log = episode(seed, profile, kind)
    np.savez_compressed(f"{out}/ep_{idx:05d}.npz", **log)
    return kind, bool(log["success"])


def collect(out, n_clean, n_fault, seed, workers=20):
    """Per profile: n_clean clean episodes, then n_fault of each fault kind; seeds disjoint from G1 (0..) and C2 (7*1e6+3)."""
    os.makedirs(out, exist_ok=True)
    jobs, i = [], 0
    for kind in [None] * n_clean + [k for k in KINDS for _ in range(n_fault)]:
        for p in PROFILES:
            jobs.append((out, i, seed * 1_000_003 + i, p, kind))
            i += 1
    with Pool(workers) as pool:
        res = pool.map(_one, jobs, chunksize=2)
    print(f"collected {len(res)} episodes, success {np.mean([r[1] for r in res]):.2f}")


def auroc(pos, neg):
    """P(score of a positive > score of a negative), ties count half."""
    x = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    for v in np.unique(x):  # average ranks over ties
        m = x == v
        r[m] = r[m].mean()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))


def score(data, ckpt, K=10, stride=3, max_blind=60):
    """-> per-segment table: episode, segment start, fault kind or 'clean', is-faulty, dyn, stale, blind."""
    import torch
    from world_model import Predictor, amp, norm, norm_xy, rollout

    dev = "cuda"
    ck = torch.load(ckpt, weights_only=False)
    H = ck["args"]["H"]
    model = Predictor(ck["n_tok"], max_t=H + ck["args"]["K"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    meta = np.load(f"{data}/meta.npz")
    emb = np.load(f"{data}/emb_p8.npy", mmap_mode="r")
    off, act = meta["offsets"], meta["action"]
    rows = []
    for e, f in enumerate(meta["files"]):
        L = np.load(f)
        z = torch.from_numpy(np.asarray(emb[off[e]:off[e + 1]])).to(dev)
        zn = norm(z)
        u = torch.from_numpy(norm_xy(act[off[e]:off[e + 1]]).astype(np.float32)).to(dev)
        n = len(zn)
        dyn, stale, blind = (np.full(n, np.nan) for _ in range(3))
        with torch.no_grad():
            a = np.arange(H - 1, n - K, stride)
            ctx = zn[torch.from_numpy(a[:, None] + np.arange(-H + 1, 1))]
            with amp():
                pred = rollout(model, ctx, u[torch.from_numpy(a[:, None] + np.arange(-H + 1, K))], K).float()
            err = (pred - zn[torch.from_numpy(a[:, None] + np.arange(1, K + 1))]).abs().mean((-1, -2)).mean(1)
            dyn[a + K // 2] = err.cpu().numpy()
            # tick t ran the command sent at e_t, which the operator chose looking at frame c = shown_tick[e_t]
            ex, sh = L["exec_tick"], L["shown_tick"]
            t = np.arange(len(ex))
            ok = ex >= 0
            c = np.full(len(ex), -1)
            c[ok] = sh[ex[ok]]
            ok &= c >= H - 1
            stale[t[ok]] = (zn[t[ok]] - zn[c[ok]]).abs().mean((-1, -2)).cpu().numpy()
            k = t - c
            for kk in np.unique(k[ok & (k >= 1) & (k <= max_blind)]):
                tt = t[ok & (k == kk)][::stride]
                cc = tt - kk
                ctx = zn[torch.from_numpy(cc[:, None] + np.arange(-H + 1, 1))]
                with amp():
                    pred = rollout(model, ctx, u[torch.from_numpy(cc[:, None] + np.arange(-H + 1, kk))], int(kk)).float()
                blind[tt] = (pred[:, -1] - zn[torch.from_numpy(tt)]).abs().mean((-1, -2)).cpu().numpy()
            blind[t[ok & (k == 0)]] = 0.0
        kind, f0, f1 = str(L["fault"]), int(L["fault_t0"]), int(L["fault_t1"])
        for s0 in range(0, n - SEG + 1, SEG):
            seg = slice(s0, s0 + SEG)
            bad = kind != "clean" and s0 < f1 and s0 + SEG > f0
            rows.append((e, s0, kind, bad, *(np.nanmax(x[seg]) if np.isfinite(x[seg]).any() else np.nan
                                              for x in (dyn, stale, blind))))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "collect", "score"])
    ap.add_argument("--data", default="data_c4pilot")
    ap.add_argument("--n-clean", type=int, default=10)
    ap.add_argument("--n-fault", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ckpt", default="data/wm_main.pt")
    args = ap.parse_args()
    if args.cmd == "check":
        assert abs(auroc(np.array([2.0, 3]), np.array([0.0, 1])) - 1) < 1e-9 and abs(auroc(np.ones(3), np.ones(4)) - 0.5) < 1e-9
        base = episode(5, "50j", None)
        for kind in KINDS:
            L = episode(5, "50j", kind)
            t0, t1 = int(L["fault_t0"]), int(L["fault_t1"])
            assert np.array_equal(L["sent"][:t0], base["sent"][:t0]), kind  # identical until the fault starts
            if kind == "hesitate":
                assert np.ptp(L["sent"][t0:t0 + 20], 0).max() == 0
            if kind == "spike":
                assert L["up"][t0:t0 + 25].min() >= 700 / 30
            if kind == "slip":
                jump = np.linalg.norm(L["state"][t0 + 1, BLOCK] - L["state"][t0, BLOCK])
                assert jump > 10, jump
        print("c4.py checks passed")
    elif args.cmd == "collect":
        collect(f"{args.data}/episodes", args.n_clean, args.n_fault, args.seed)
    else:
        rows = score(args.data, args.ckpt)
        kinds = np.array([r[2] for r in rows])
        bad = np.array([r[3] for r in rows])
        S = np.array([r[4:] for r in rows], float)
        np.savez(f"{args.data}/scores.npz", ep=[r[0] for r in rows], s0=[r[1] for r in rows], kind=kinds, bad=bad, S=S)
        neg = kinds == "clean"
        print(f"{len(rows)} segments; AUROC of each score, faulty segments of a kind vs clean-episode segments")
        print(f"{'kind':>9} {'n':>5} " + " ".join(f"{s:>7}" for s in ["dyn", "stale", "blind"]))
        for k in KINDS:
            pos = (kinds == k) & bad
            print(f"{k:>9} {pos.sum():5d} " + " ".join(
                f"{auroc(S[pos, j][~np.isnan(S[pos, j])], S[neg, j][~np.isnan(S[neg, j])]):7.3f}" for j in range(3)))
