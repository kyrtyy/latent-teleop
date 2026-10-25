"""README demo from one C2 scene under 300 ms jittery delay (a scene where the ghost run succeeds and the
no-ghost run does not; any seed works).
  assets/ghost.gif     what the operator receives (old frame) | what the learned ghost predicts | what is really there
  assets/compare.gif   the robot's real scene: no ghost | Smith predictor | learned ghost, same scene, operator, delays
  assets/demo.mp4      both, at real time (one frame per 30 ms tick), with title and caption cards (mp4v; re-encode
                       to H.264 for browsers: ffmpeg -i demo.mp4 -c:v libx264 -pix_fmt yuv420p -crf 23 out.mp4)
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from c2 import learned_cell, setup
from sim import BLOCK, make_env, render
from teleop import run_episode

S, GAP, HEAD, FOOT = 320, 8, 64, 40  # panel px, gap, header, footer (coverage bar)
INK, MUTED, BG, LINE = (18, 28, 43), (90, 100, 115), (244, 241, 234), (210, 204, 192)
ARM = {"none": (142, 154, 171), "kinematic": (47, 111, 176), "learned": (180, 80, 31), "real": (18, 28, 43)}


def font(size, bold=False):
    for f in [f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
              f"/Library/Fonts/Arial{' Bold' if bold else ''}.ttf"]:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size)


F_T, F_S, F_BIG, F_MID = font(20, True), font(16), font(40, True), font(22)


def panel(env, s, title, sub, accent, cov=None, done=False):
    out = Image.new("RGB", (S, HEAD + S + FOOT), BG)
    out.paste(Image.fromarray(render(env, s)).convert("RGB").resize((S, S), Image.LANCZOS), (0, HEAD))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, 5, HEAD - 8], fill=accent)
    d.text((14, 6), title, font=F_T, fill=INK)
    d.text((14, 34), sub, font=F_S, fill=MUTED)
    if cov is not None:  # coverage bar: share of the target the T covers, label right of the bar
        y = HEAD + S + 14
        d.rectangle([0, y, S - 101, y + 14], fill=LINE)
        d.rectangle([0, y, int((S - 101) * min(cov, 1)), y + 14], fill=accent)
        d.text((S - 92, y - 3), f"{cov:.0%}" + (" done" if done else ""), font=F_S, fill=INK)
    return out


def row(panels):
    out = Image.new("RGB", (len(panels) * (S + GAP) - GAP, panels[0].height), BG)
    for i, p in enumerate(panels):
        out.paste(p, (i * (S + GAP), 0))
    return out


def card(size, title, lines):
    out = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(out)
    d.text((40, size[1] // 2 - 30 - 18 * len(lines)), title, font=F_BIG, fill=INK)
    for i, l in enumerate(lines):
        d.text((40, size[1] // 2 + 38 + 34 * i - 18 * len(lines)), l, font=F_MID, fill=MUTED)
    return out


def captioned(frame, size, text):
    out = Image.new("RGB", size, BG)
    out.paste(frame, ((size[0] - frame.width) // 2, 0))
    ImageDraw.Draw(out).text((size[0] // 2, size[1] - 36), text, font=F_MID, fill=MUTED, anchor="ma")
    return out


def save_gif(frames, f, ms):
    frames[0].save(f, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    print(f"wrote {f} ({len(frames)} frames, {os.path.getsize(f) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7000070)
    ap.add_argument("--profile", default="300j")
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--every", type=int, default=2)  # GIF: 2 ticks = 60 ms per frame, real time
    ap.add_argument("--ckpt", default="data/wm_main.pt")
    ap.add_argument("--probe-ckpt", default="data_big/probe150k_p8.pt")
    args = ap.parse_args()
    os.makedirs("assets", exist_ok=True)
    G = learned_cell([args.seed], args.profile, args.ticks, args.ckpt, args.probe_ckpt, raw=True)[0]
    N = run_episode(*setup(args.seed, args.profile, "none")[:4], args.ticks)
    env, op, up, down, g = setup(args.seed, args.profile, "kinematic")
    K = run_episode(env, op, up, down, args.ticks, g)
    env = make_env()

    def ghost_frame(t):
        n = len(G["exec"])
        T, c = min(int(G["ghost_tick"][t]), n), max(int(G["shown_tick"][t]), 0)
        err = np.linalg.norm(G["perceived"][t][BLOCK] - G["state"][T][BLOCK])
        return row([panel(env, G["state"][c], "Operator receives", f"a frame {(t - c) * 30} ms old", ARM["none"]),
                    panel(env, G["perceived"][t], "Learned ghost", f"predicts {(T - c) * 30} ms ahead · block off {err:.0f} px", ARM["learned"]),
                    panel(env, G["state"][T], "Reality", "at the predicted moment", ARM["real"])])

    runs = [(N, "No ghost", "acts on the old frame", "none"), (K, "Smith predictor", "robot replayed, block held", "kinematic"),
            (G, "Learned ghost", "world-model prediction", "learned")]
    n_cmp = max(len(L["exec"]) for L, *_ in runs) + 30

    def compare_frame(t):
        ps = []
        for L, name, sub, arm in runs:
            i = min(t, len(L["exec"]))
            ps.append(panel(env, L["state"][i], f"{name}  {i * 0.03:4.1f} s", sub, ARM[arm], L["coverage"][i],
                            done=t >= len(L["exec"]) and bool(L["success"])))
        return row(ps)

    save_gif([ghost_frame(t) for t in range(0, len(G["exec"]), args.every)], "assets/ghost.gif", args.every * 30)
    save_gif([compare_frame(t) for t in range(0, n_cmp, args.every)], "assets/compare.gif", args.every * 30)

    import cv2
    first = compare_frame(0)
    size = (first.width + 80, first.height + 64)
    size = (size[0] + size[0] % 2, size[1] + size[1] % 2)
    fps = 1000 / 30
    vw = cv2.VideoWriter("assets/demo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    put = lambda im, n=1: [vw.write(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)) for _ in range(n)]
    put(card(size, "Seeing ahead through network delay", [
        "A robot pushes a T-shaped block onto its target, driven by an operator over a network",
        f"with {args.profile[:-1]} ms jittery delay each way. A world model predicts what the robot sees now.",
        "PushT simulation · scripted operator · real time"]), int(4 * fps))
    put(card(size, "1 · What the operator sees", ["Left: the delayed frame. Middle: the ghost the operator acts on.",
                                                 "Right: what is actually there at the moment the ghost predicts."]), int(3 * fps))
    for t in range(len(G["exec"])):
        put(captioned(ghost_frame(t), size, "The delayed frame shows where the block was; the ghost shows where it is going"))
    put(card(size, "2 · Same scene, same operator, same delays", ["Only the display differs. Bars show how much of the target",
                                                                "the block covers; 95% counts as done."]), int(3 * fps))
    for t in range(n_cmp):
        put(captioned(compare_frame(t), size, "Only the display differs between the three runs"))
    put(card(size, "Across 200 scenes at 300 ms", ["Learned ghost recovers 90% of the performance delay takes away",
                                                  "Smith predictor recovers 59% · task success 4% → 20% with the ghost",
                                                  "github.com/kyrtyy/latent-teleop"]), int(5 * fps))
    vw.release()
    print(f"wrote assets/demo.mp4 ({os.path.getsize('assets/demo.mp4') / 1e6:.1f} MB)")
