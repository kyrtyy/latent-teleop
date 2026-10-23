"""C4 self-auditing: inject labelled faults into teleoperated demonstrations, then score every segment.

Faults (one per faulty episode, window [t0, t1) in ticks; `effect` extends it by how long the fault shows):
  hesitate  the operator freezes the cursor for 20-40 ticks
  wrong     the operator pushes toward a random point for 15-30 ticks (a mistaken push)
  spike     both links get +700 ms on every packet sent during 25 ticks (a network spike)
  slip      the block jumps 15-30 px and 0.15-0.3 rad away from the agent (an unexpected contact / slip)
Natural hesitations are switched off (op hesitate = 0), so clean episodes are clean.

Segment scores (SEG ticks each), from frozen DINOv2 tokens and the C1 world model:
  dyn    k-step latent prediction error from the true context under the executed commands (physics surprise)
  age, stall  log rules (declared secondary): oldest command run in the segment, longest hold of one command
The pilot also tried `stale` and `blind` (operator-view scores; README); they never beat dyn and were removed.
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
    out, idx, seed, profile, kind, max_ticks = job
    log = episode(seed, profile, kind, max_ticks)
    np.savez_compressed(f"{out}/ep_{idx:05d}.npz", **log)
    return kind, bool(log["success"])


def collect(out, n_clean, n_fault, seed, max_ticks=600, workers=20):
    """Per profile: n_clean clean episodes, then n_fault of each fault kind; seeds disjoint from G1 (0..) and C2 (7*1e6+3)."""
    os.makedirs(out, exist_ok=True)
    jobs, i = [], 0
    for kind in [None] * n_clean + [k for k in KINDS for _ in range(n_fault)]:
        for p in PROFILES:
            jobs.append((out, i, seed * 1_000_003 + i, p, kind, max_ticks))
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


def score(data, ckpt, K=10, stride=3):
    """-> per-segment table: episode, segment index, profile, fault kind or 'clean', is-faulty, dyn, age, stall.
    dyn: world-model k-step latent error under the executed commands; each anchor's error is credited to every tick
    of its predicted window, and a segment takes the max. age, stall (log rules, no model): the oldest command the
    robot ran in the segment (ticks), and the longest run of ticks it held one command."""
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
        zn = norm(torch.from_numpy(np.array(emb[off[e]:off[e + 1]])).to(dev))
        u = torch.from_numpy(norm_xy(act[off[e]:off[e + 1]]).astype(np.float32)).to(dev)
        T = len(L["exec"])
        dyn = np.full(T + 1, -np.inf)
        a = np.arange(H - 1, T + 1 - K, stride)
        if len(a):
            with torch.no_grad(), amp():
                pred = rollout(model, zn[torch.from_numpy(a[:, None] + np.arange(-H + 1, 1))],
                               u[torch.from_numpy(a[:, None] + np.arange(-H + 1, K))], K).float()
            err = (pred - zn[torch.from_numpy(a[:, None] + np.arange(1, K + 1))]).abs().mean((-1, -2)).mean(1).cpu().numpy()
            np.maximum.at(dyn, a[:, None] + np.arange(1, K + 1), np.repeat(err[:, None], K, 1))
        dyn = dyn[1:]  # tick t's command produced frame t+1
        ex = L["exec_tick"]
        age = np.where(ex >= 0, np.arange(T) - ex, np.arange(T) + 1).astype(float)
        same = np.r_[False, np.all(np.diff(L["exec"], axis=0) == 0, 1)]
        run = np.zeros(T)
        for t in range(1, T):
            run[t] = run[t - 1] + 1 if same[t] else 0
        kind, f0, f1 = str(L["fault"]), int(L["fault_t0"]), int(L["fault_t1"])
        for i in range((T + SEG - 1) // SEG):
            seg = slice(i * SEG, (i + 1) * SEG)
            bad = kind != "clean" and i * SEG < f1 and (i + 1) * SEG > f0
            d = dyn[seg]
            rows.append((e, i, str(L["profile"]), kind, bad, d.max() if np.isfinite(d).any() else np.nan,
                         age[seg].max(), run[seg].max()))
    return rows


def table(rows):
    return {k: np.array([r[j] for r in rows]) for j, k in enumerate(["ep", "seg", "profile", "kind", "bad", "dyn", "age", "stall"])}


def thresholds(cal, key, alpha):
    """Split conformal per delay profile: the ceil((n+1)(1-alpha))-th smallest clean calibration score."""
    out = {}
    for p in np.unique(cal["profile"]):
        x = np.sort(cal[key][(cal["profile"] == p) & ~np.isnan(cal[key])])
        out[p] = x[min(len(x) - 1, int(np.ceil((len(x) + 1) * (1 - alpha))) - 1)]
    return out


def flag(t, thr, key):
    return np.array([v > thr[p] for v, p in zip(t[key], t["profile"])])


def flags(data, cal_data, hold_data, ckpt, alpha=0.05):
    cal, hold, rq = (table(score(d, ckpt)) for d in (cal_data, hold_data, data))
    np.savez(f"{data}/scores.npz", **rq)
    th = {k: thresholds(cal, k, alpha if k == "dyn" else alpha / 2) for k in ("dyn", "age", "stall")}
    out = {"wm": flag(rq, th["dyn"], "dyn"), "rules": flag(rq, th["age"], "age") | flag(rq, th["stall"], "stall")}
    per_ep = {a: [m[rq["ep"] == e] for e in range(rq["ep"].max() + 1)] for a, m in out.items()}
    np.savez(f"{data}/flags.npz", **{a: np.array(v, dtype=object) for a, v in per_ep.items()})
    print(f"alpha {alpha}: false-alarm rate on held-out clean segments: wm "
          f"{flag(hold, th['dyn'], 'dyn').mean():.3f}, rules {(flag(hold, th['age'], 'age') | flag(hold, th['stall'], 'stall')).mean():.3f}")
    clean = rq["kind"] == "clean"
    print(f"{'kind':>9} {'segs':>5} | AUROC within profile: {'dyn':>6} {'age':>6} {'stall':>6} | "
          f"wm recall / event recall | rules recall / event recall")
    for k in KINDS:
        pos = (rq["kind"] == k) & rq["bad"]
        au = []
        for key in ("dyn", "age", "stall"):
            v = [auroc(rq[key][pos & (rq["profile"] == p)], rq[key][clean & (rq["profile"] == p)])
                 for p in np.unique(rq["profile"])]
            au.append(np.nanmean(v))
        ev = {a: np.mean([m[pos & (rq["ep"] == e)].any() for e in np.unique(rq["ep"][pos])]) for a, m in out.items()}
        print(f"{k:>9} {pos.sum():5d} | {'':>21} {au[0]:6.3f} {au[1]:6.3f} {au[2]:6.3f} | "
              f"{out['wm'][pos].mean():.2f} / {ev['wm']:.2f}             | {out['rules'][pos].mean():.2f} / {ev['rules']:.2f}")
    for a, m in out.items():
        print(f"{a}: flags {m.sum()} segments ({m.mean():.1%}); precision vs injected faults {rq['bad'][m].mean():.2f}; "
              f"fault segments flagged {m[rq['bad']].mean():.2f}; clean-episode segments flagged {m[clean].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "collect", "flags"])
    ap.add_argument("--data", default="data_c4pilot")
    ap.add_argument("--n-clean", type=int, default=10)
    ap.add_argument("--n-fault", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ckpt", default="data/wm_main.pt")
    ap.add_argument("--max-ticks", type=int, default=600)
    args = ap.parse_args()
    if args.cmd == "check":
        assert abs(auroc(np.array([2.0, 3]), np.array([0.0, 1])) - 1) < 1e-9 and abs(auroc(np.ones(3), np.ones(4)) - 0.5) < 1e-9
        cal = {"profile": np.array(["a"] * 100), "x": np.arange(100.0)}
        assert thresholds(cal, "x", 0.05)["a"] == 95  # ceil(101 * 0.95) = 96th smallest
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
        collect(f"{args.data}/episodes", args.n_clean, args.n_fault, args.seed, args.max_ticks)
    else:
        flags(args.data, args.data + "cal", args.data + "hold", args.ckpt)
