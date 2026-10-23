"""C3 intent-preserving execution. Every tick the operator sends its intended path (the next N cursor targets it
would send if the scene stayed as it now looks), not one setpoint. The robot keeps executing the newest path while
packets are late, blends into a newly landed path after a gap (real-time-chunking style), and can run a safety layer:
while extrapolating on its own, it stops before a move the model predicts would push the block. The robot never
picks a new goal; contact needs a fresh command from the operator.

Arms: setpoint (the C2 loop: run the newest command, hold otherwise), chunk (hard switch), blend, and blend plus a
safety layer: `wm` (world model on the robot's live frames) or `oracle` (true physics).
Writes <out>/<profile>_<arm>.npz per cell; N paired episodes, scripted operator with natural hesitations off.
"""
import argparse
import copy
import os
from multiprocessing import Pool

import numpy as np

from sim import AGENT, ANGLE, BLOCK, ScriptedOperator, full_state, make_env, set_state
from teleop import Delay, Link, run_episode

N, BLEND, K_SAFE = 10, 5, 10  # path length (300 ms), blend ticks after a gap, safety look-ahead (ticks)
ARMS = ["setpoint", "chunk", "blend", "wm", "oracle"]


class PathOperator:
    """Sends [this tick's command, then what it would send over the next N-1 ticks on the same view]."""

    def __init__(self, op):
        self.op, self.params = op, op.params

    def act(self, s, t):
        path = [self.op.act(s, t)]
        twin = copy.deepcopy(self.op)  # the real operator's state and random stream are untouched
        path += [twin.act(s, t + i) for i in range(1, N)]
        return np.stack(path)


class ChunkExecutor:
    def __init__(self, env, start, blend, safety=None):
        self.env, self.blend, self.safety = env, blend, safety
        self.e, self.path, self.land, self.old, self.cmd = -1, None, 0, None, np.asarray(start, float)
        self.j, self.stopped = [], []  # per tick: path index executed (0 = fresh), safety stop
        self.states, self.hist = [], []  # per tick: the robot's live state, the command it executed

    def __call__(self, t, arrived):
        self.states.append(full_state(self.env))
        for sent_t, p in arrived:
            if sent_t > self.e:
                gap = self.path is not None and t - self.land >= 2  # the old path was being extrapolated
                self.old = self.path[np.minimum(np.arange(t - self.land, t - self.land + BLEND), N - 1)] \
                    if self.blend and gap else None
                self.e, self.path, self.land = sent_t, p, t
        if self.path is None:
            self.j.append(-1), self.stopped.append(False), self.hist.append(self.cmd.copy())
            return -1, self.cmd
        j = t - self.land
        cmd = self.path[min(j, N - 1)]
        if self.old is not None and j < BLEND:
            w = (j + 1) / (BLEND + 1)
            cmd = w * cmd + (1 - w) * self.old[j]
        stop = j >= 1 and self.safety is not None and self.safety(self, t, self.path[np.minimum(np.arange(j, j + K_SAFE), N - 1)])
        if not stop:
            self.cmd = cmd
        self.j.append(j), self.stopped.append(stop), self.hist.append(self.cmd.copy())
        return self.e, self.cmd.copy()


def moves(b_path, b_hold, th_path, th_hold):
    """Would following the path move the block more than stopping (3 px or 0.05 rad)?"""
    return bool(np.linalg.norm(b_path - b_hold) > 3 or abs(th_path - th_hold) > 0.05)


class OracleSafety:
    """True physics from the robot's live state: would the remaining path move the block?"""

    def __init__(self):
        self.sim = make_env()

    def __call__(self, ex, t, path):
        end = []
        for p in (path, np.repeat(ex.cmd[None], len(path), 0)):  # the remaining path vs holding where it is
            set_state(self.sim, ex.states[t])
            for c in p:
                self.sim.step(c)
            end.append(full_state(self.sim))
        return moves(end[0][BLOCK], end[1][BLOCK], end[0][ANGLE], end[1][ANGLE])


class WMSafety:
    """The C1 world model on the robot's own (undelayed) frames and the commands it executed: roll the remaining
    path forward and read the block out with the probe. Runs on CPU next to the robot loop."""

    def __init__(self, model, probe, enc, H):
        self.model, self.probe, self.enc, self.H = model, probe, enc, H
        self.renv, self.tok = make_env(), {}

    def __call__(self, ex, t, path):
        import torch
        from embed import encode
        from sim import render
        from world_model import decode, norm, norm_xy, rollout
        ctx = list(range(t - self.H + 1, t + 1))
        if ctx[0] < 0:
            return False
        new = [c for c in ctx if c not in self.tok]
        if new:
            img = torch.from_numpy(np.stack([render(self.renv, ex.states[c]) for c in new]))
            self.tok.update(zip(new, encode(self.enc, img)))
        z = norm(torch.stack([self.tok[c] for c in ctx]))[None].repeat(2, 1, 1, 1)
        past = [ex.hist[c] for c in ctx[:-1]]
        a = np.stack([np.concatenate([past, p]) for p in (path, np.repeat(ex.cmd[None], len(path), 0))])
        with torch.no_grad():  # differential: the probe's own bias cancels between the two rollouts
            pred = rollout(self.model, z, torch.from_numpy(norm_xy(a).astype(np.float32)), len(path))
            _, b, th, _ = decode(self.probe(pred[:, -1]))
        return moves(b[0].numpy(), b[1].numpy(), float(th[0]), float(th[1]))


def run(seed, profile, arm, max_ticks, wm=None):
    op_rng, up_rng, down_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))
    env = make_env(seed)
    op = ScriptedOperator(op_rng, full_state(env)[AGENT])
    op.params["hesitate"] = 0.0
    up, down = Link(Delay(profile, up_rng)), Link(Delay(profile, down_rng))
    if arm == "setpoint":
        L, ex = run_episode(env, op, up, down, max_ticks), None
    else:
        safety = {"wm": lambda: WMSafety(*wm), "oracle": OracleSafety}.get(arm, lambda: None)()
        ex = ChunkExecutor(env, full_state(env)[AGENT], blend=arm != "chunk", safety=safety)
        L = run_episode(env, PathOperator(op), up, down, max_ticks, executor=ex)
    return summarize(L, ex, max_ticks)


def summarize(L, ex, max_ticks):
    cov = np.full(max_ticks + 1, L["coverage"][-1])
    cov[:len(L["coverage"])] = L["coverage"]
    E, A = L["exec"], L["state"][:, AGENT]
    held = np.r_[False, np.all(np.abs(np.diff(E, axis=0)) < 1e-9, 1)]
    runs, n = [], 0
    for h in held:
        n = n + 1 if h else 0
        if n == 3:
            runs.append(1)  # a pause: the robot's setpoint frozen for 3+ ticks (operator hesitations are off)
    minutes = len(E) * 0.03 / 60
    jerk = np.linalg.norm(np.diff(A, 3, axis=0), axis=-1).mean() if len(A) > 3 else np.nan
    j = np.array(ex.j) if ex is not None else np.zeros(len(E))
    moved = np.linalg.norm(np.diff(L["state"][:, BLOCK], axis=0), axis=-1)
    robot_push = moved[j >= 1].sum()  # block motion (px) while the robot runs on its own, without a fresh command
    return dict(coverage=cov, success=bool(L["success"]), pauses_per_min=len(runs) / minutes, jerk=jerk,
                held_frac=held.mean(), robot_push=robot_push, extrap_frac=(j >= 1).mean(),
                stops_per_min=(np.sum(ex.stopped) if ex is not None else 0) / minutes)


def _job(job):
    seed, profile, arm, max_ticks = job
    wm = None
    if arm == "wm":
        wm = _load_wm()
    return run(seed, profile, arm, max_ticks, wm)


_WM = None


def _load_wm(ckpt="data/wm_main.pt", probe_ckpt="data_big/probe150k_p8.pt"):
    global _WM
    if _WM is None:
        import torch
        from embed import encoder
        from world_model import Predictor, Probe
        torch.set_num_threads(2)
        ck = torch.load(ckpt, weights_only=False, map_location="cpu")
        model = Predictor(ck["n_tok"], max_t=ck["args"]["H"] + ck["args"]["K"]).eval()
        model.load_state_dict(ck["model"])
        probe = Probe(ck["n_tok"]).eval()
        probe.load_state_dict(torch.load(probe_ckpt, weights_only=False, map_location="cpu")["probe"])
        _WM = (model, probe, encoder("cpu"), ck["args"]["H"])
    return _WM


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run"])
    ap.add_argument("--profiles", nargs="+", default=["50j", "150j", "150b", "300j", "300b"])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-ticks", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--out", default="results/c3")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    if args.cmd == "check":
        # with no delay, or a constant one, a path lands every tick and the robot runs its first element: same episode
        for prof in ["0"]:
            a, b = run(3, prof, "setpoint", 300), run(3, prof, "blend", 300)
            assert np.allclose(a["coverage"], b["coverage"]) and b["extrap_frac"] == 0, (a["coverage"][-1], b["coverage"][-1])
        # under bursty jitter the robot keeps moving: fewer pauses than holding the last setpoint
        a, b = (np.mean([run(s, "300b", arm, 400)["pauses_per_min"] for s in range(6)]) for arm in ["setpoint", "blend"])
        assert b < a, (a, b)
        o = run(4, "300b", "oracle", 400)
        print(f"c3.py checks passed (300b pauses/min: setpoint {a:.1f}, blend {b:.1f}; oracle stops/min {o['stops_per_min']:.1f})")
    else:
        os.makedirs(args.out, exist_ok=True)
        seeds = [args.seed * 1_000_003 + i for i in range(args.n)]
        for profile in args.profiles:
            for arm in args.arms:
                f = f"{args.out}/{profile}_{arm}.npz"
                if os.path.exists(f):
                    continue
                with Pool(args.workers) as pool:
                    res = pool.map(_job, [(s, profile, arm, args.max_ticks) for s in seeds], chunksize=1)
                np.savez(f, seeds=seeds, **{k: np.array([r[k] for r in res]) for k in res[0]})
                print(f"{profile:>5} {arm:>8}: AUC {np.mean([r['coverage'].mean() for r in res]):.3f} "
                      f"success {np.mean([r['success'] for r in res]):.2f} pauses/min {np.mean([r['pauses_per_min'] for r in res]):5.1f} "
                      f"jerk {np.nanmean([r['jerk'] for r in res]):.2f} robot-push {np.mean([r['robot_push'] for r in res]):6.1f}px "
                      f"stops/min {np.mean([r['stops_per_min'] for r in res]):5.1f}", flush=True)
