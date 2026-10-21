"""Record scripted-operator episodes under each delay profile. Logs states only; frames are re-rendered by embed.py."""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from sim import AGENT, ScriptedOperator, full_state, make_env
from teleop import PROFILES, Delay, Link, run_episode


def one(job):
    out, idx, profile, seed, max_ticks = job
    env, rng = make_env(seed), np.random.default_rng(seed)
    op = ScriptedOperator(rng, full_state(env)[AGENT])
    log = run_episode(env, op, Link(Delay(profile, rng)), Link(Delay(profile, rng)), max_ticks)
    np.savez_compressed(f"{out}/ep_{idx:05d}.npz", profile=profile, seed=seed, **log,
                        **{f"op_{k}": v for k, v in op.params.items()})
    return profile, bool(log["success"]), len(log["exec"]), log["contact"].mean()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/episodes")
    ap.add_argument("--per-profile", type=int, default=200)
    ap.add_argument("--profiles", nargs="+", default=list(PROFILES))
    ap.add_argument("--max-ticks", type=int, default=600)  # 18 s
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # episodes interleave profiles so any prefix covers every profile; world_model.episode_split relies on this order
    jobs = [(args.out, i * len(args.profiles) + j, p, args.seed * 1_000_003 + i * len(args.profiles) + j, args.max_ticks)
            for i in range(args.per_profile) for j, p in enumerate(args.profiles)]
    with Pool(args.workers) as pool:
        res = pool.map(one, jobs, chunksize=4)
    print(f"{'profile':>8} {'success':>8} {'len':>6} {'contact':>8}")
    for p in args.profiles:
        r = [x for x in res if x[0] == p]
        print(f"{p:>8} {np.mean([x[1] for x in r]):8.2f} {np.mean([x[2] for x in r]):6.0f} {np.mean([x[3] for x in r]):8.2f}")
