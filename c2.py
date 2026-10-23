"""C2 closed-loop experiment: the scripted operator acts on a ghost under each delay profile.
One cell = (profile, ghost kind) x N episodes with paired seeds: the same scene, operator parameters and delay
sequences in every cell (operator, uplink and downlink draw from separate seeded streams).
none/kinematic/oracle run on CPU workers. learned runs all N episodes in lockstep, with one batched DINOv2 encode
and one batched world-model rollout per tick.
Writes <out>/<profile>_<kind>.npz: per-episode coverage curve, success, length, ghost error, ghost stats."""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from ghost import Ghost, H
from sim import AGENT, ANGLE, BLOCK, ScriptedOperator, full_state, make_env, wrap
from teleop import Delay, Link, episode, run_episode


def setup(seed, profile, kind):
    op_rng, up_rng, down_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))
    env = make_env(seed)
    g = Ghost(kind)
    return env, ScriptedOperator(op_rng, full_state(env)[AGENT]), Link(Delay(profile, up_rng)), Link(Delay(profile, down_rng)), g


def summarize(L, g, max_ticks):
    cov = np.full(max_ticks + 1, L["coverage"][-1])  # after success the scene stays solved
    cov[:len(L["coverage"])] = L["coverage"]
    n = len(L["state"])
    T = np.arange(len(L["up"])) + L["up"]  # the tick this command actually lands: score every ghost against it
    ok = T < n
    gb = np.linalg.norm(L["perceived"][ok, BLOCK] - L["state"][T[ok], BLOCK], axis=-1)
    ga = np.degrees(np.abs(wrap(L["perceived"][ok, ANGLE] - L["state"][T[ok], ANGLE])))
    moving = np.linalg.norm(L["state"][T[ok], BLOCK] - L["state"][np.maximum(L["shown_tick"][ok], 0), BLOCK], axis=-1) >= 5
    return dict(coverage=cov, success=bool(L["success"]), length=len(L["exec"]), contact=L["contact"].mean(),
                ghost_block_px=gb.mean(), ghost_angle_deg=ga.mean(),
                ghost_block_px_moving=gb[moving].mean() if moving.any() else np.nan,
                **{f"n_{k}": v for k, v in g.stats.items()})


def cpu_one(job):
    seed, profile, kind, max_ticks = job
    env, op, up, down, g = setup(seed, profile, kind)
    return summarize(run_episode(env, op, up, down, max_ticks, g), g, max_ticks)


def learned_cell(seeds, profile, max_ticks, ckpt, probe_ckpt, chunk=16):  # 64 OOMs a 6 GB GPU at 300 ms
    import torch
    from embed import encode, encoder
    from sim import render
    from world_model import Predictor, Probe, amp, norm, norm_xy, rollout

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, weights_only=False, map_location=dev)
    n_tok = ck["n_tok"]
    model = Predictor(n_tok, max_t=ck["args"]["H"] + ck["args"]["K"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    probe = Probe(n_tok).to(dev).eval()
    probe.load_state_dict(torch.load(probe_ckpt, weights_only=False, map_location=dev)["probe"])
    enc, renv = encoder(dev), make_env()
    assert ck["args"]["H"] == H

    eps = [setup(s, profile, "learned") for s in seeds]
    gens = [episode(env, op, up, down, max_ticks, g) for env, op, up, down, g in eps]
    cache = [{} for _ in eps]  # per episode: frame tick -> tokens (fp16, on GPU)
    logs, reqs = [None] * len(eps), {}

    def advance(i, answer):
        try:
            reqs[i] = gens[i].send(answer)
        except StopIteration as done:
            logs[i] = done.value

    for i in range(len(eps)):
        advance(i, None)
    t0, step = time.time(), 0
    with torch.no_grad():
        while reqs:
            batch, reqs_now = list(reqs.items()), {}
            reqs.clear()
            todo = [(i, c, s) for i, r in batch for c, s in zip(r["ctx"], r["states"]) if c not in cache[i]]
            for j in range(0, len(todo), 256):
                part = todo[j:j + 256]
                tok = encode(enc, torch.from_numpy(np.stack([render(renv, s) for _, _, s in part])))
                for (i, c, _), z in zip(part, tok):
                    cache[i][c] = z
            for i, r in batch:  # drop frames no future context can use
                for c in [c for c in cache[i] if c < r["ctx"][0] - 64]:
                    del cache[i][c]
            batch.sort(key=lambda x: x[1]["k"])
            answers = {}
            for j in range(0, len(batch), chunk):
                part = batch[j:j + chunk]
                k = max(r["k"] for _, r in part)
                z = norm(torch.stack([torch.stack([cache[i][c] for c in r["ctx"]]) for i, r in part]))
                a = np.stack([np.concatenate([r["actions"], np.repeat(r["actions"][-1:], H - 1 + k - len(r["actions"]), 0)])
                              for _, r in part])
                with amp():
                    pred = rollout(model, z, torch.from_numpy(norm_xy(a).astype(np.float32)).to(dev), k).float()
                last = pred[torch.arange(len(part)), torch.tensor([r["k"] - 1 for _, r in part])]
                for (i, _), p in zip(part, probe(last).float().cpu().numpy()):
                    answers[i] = p
            for i, _ in batch:
                advance(i, answers[i])
            step += 1
            if step % 200 == 0:
                print(f"  {profile} learned: {step} batched steps, {len(reqs)} waiting, {time.time() - t0:.0f}s", flush=True)
    return [summarize(L, e[4], max_ticks) for L, e in zip(logs, eps)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", default=["50j", "150j", "150b", "300j", "300b"])
    ap.add_argument("--kinds", nargs="+", default=["none", "kinematic", "oracle", "learned"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-ticks", type=int, default=1000)  # 30 s
    ap.add_argument("--seed", type=int, default=7)  # disjoint from the training data's seeds
    ap.add_argument("--ckpt", default="data/wm_main.pt")
    ap.add_argument("--probe-ckpt", default="data_big/probe150k_p8.pt")
    ap.add_argument("--out", default="results/c2")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = [args.seed * 1_000_003 + i for i in range(args.n)]
    for profile in args.profiles:
        for kind in args.kinds:
            f = f"{args.out}/{profile}_{kind}.npz"
            if os.path.exists(f):
                continue
            t0 = time.time()
            if kind == "learned":
                res = learned_cell(seeds, profile, args.max_ticks, args.ckpt, args.probe_ckpt)
            else:
                with Pool(args.workers) as pool:
                    res = pool.map(cpu_one, [(s, profile, kind, args.max_ticks) for s in seeds], chunksize=2)
            np.savez(f, seeds=seeds, **{k: np.array([r[k] for r in res]) for k in res[0]})
            print(f"{profile:>5} {kind:>9}: coverage AUC {np.mean([r['coverage'].mean() for r in res]):.3f}  "
                  f"success {np.mean([r['success'] for r in res]):.2f}  "
                  f"ghost block err {np.nanmean([r['ghost_block_px'] for r in res]):.1f}px  ({time.time() - t0:.0f}s)", flush=True)
