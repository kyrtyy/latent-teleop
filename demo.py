"""README demo GIFs from one C2 scene under 300 ms jittery delay (a scene where the ghost run succeeds and the
no-ghost run does not; any seed works).
  assets/ghost.gif     what the operator receives (old frame) | what the learned ghost predicts | what is really there
  assets/compare.gif   the robot's real scene: no ghost vs learned ghost, same scene, operator and delays
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

from c2 import learned_cell, setup
from sim import make_env, render
from teleop import run_episode


def panel(env, s, title, sub=""):
    im = Image.fromarray(render(env, s)).convert("RGB")
    out = Image.new("RGB", (im.width, im.height + 34), "white")
    out.paste(im, (0, 34))
    d = ImageDraw.Draw(out)
    d.text((6, 3), title, fill="black")
    d.text((6, 18), sub, fill=(90, 90, 90))
    return out


def row(panels):
    out = Image.new("RGB", (sum(p.width for p in panels) + 6 * (len(panels) - 1), panels[0].height), (200, 200, 200))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + 6
    return out


def save(frames, f, ms):
    frames[0].save(f, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    print(f"wrote {f} ({len(frames)} frames, {os.path.getsize(f) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7000070)
    ap.add_argument("--profile", default="300j")
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--every", type=int, default=2)  # 2 ticks = 60 ms per GIF frame: real time
    ap.add_argument("--ckpt", default="data/wm_main.pt")
    ap.add_argument("--probe-ckpt", default="data_big/probe150k_p8.pt")
    args = ap.parse_args()
    os.makedirs("assets", exist_ok=True)
    G = learned_cell([args.seed], args.profile, args.ticks, args.ckpt, args.probe_ckpt, raw=True)[0]
    N = run_episode(*setup(args.seed, args.profile, "none")[:4], args.ticks)
    env = make_env()
    n_g, n_n = len(G["exec"]), len(N["exec"])

    frames = []
    for t in range(0, n_g, args.every):
        T = min(int(G["ghost_tick"][t]), n_g)  # the tick the ghost is predicting
        old = G["state"][max(int(G["shown_tick"][t]), 0)]
        age = (t - max(int(G["shown_tick"][t]), 0)) * 30
        frames.append(row([panel(env, old, "Operator receives", f"frame {age} ms old"),
                           panel(env, G["perceived"][t], "Learned ghost", f"predicts {(T - t) * 30:+d} ms ahead"),
                           panel(env, G["state"][T], "Reality", "at the predicted moment")]))
    save(frames, "assets/ghost.gif", args.every * 30)

    frames = []
    for t in range(0, max(n_g, n_n) + 30, args.every):
        a, b = N["state"][min(t, n_n)], G["state"][min(t, n_g)]
        ca, cb = N["coverage"][min(t, n_n)], G["coverage"][min(t, n_g)]
        frames.append(row([panel(env, a, f"No ghost  t={min(t, n_n) * 0.03:4.1f}s", f"coverage {ca:.0%}" + ("  DONE" if t >= n_n and N["success"] else "")),
                           panel(env, b, f"Learned ghost  t={min(t, n_g) * 0.03:4.1f}s", f"coverage {cb:.0%}" + ("  DONE" if t >= n_g and G["success"] else ""))]))
    save(frames, "assets/compare.gif", args.every * 30)
