"""G1 gate: how far ahead can the scene be predicted? Error vs horizon on test episodes, learned model vs
hold-last-frame, constant velocity and the kinematic (Smith) predictor, split into contact and free-space windows.

Learned model is fed the true executed commands, so this is the model's upper bound (delay estimation comes in C2).
Baselines are run twice: on the true delayed state ("true") and on the probe's reading of that frame ("probe"),
so prediction error is separable from perception error."""
import argparse
import csv
import json
import os

import numpy as np
import torch

from sim import AGENT, ANGLE, BLOCK, TICK_MS, agent_pd, wrap
from world_model import amp, episode_split, Predictor, Probe, decode, norm, norm_xy, rollout

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--pool", type=int, default=8)
ap.add_argument("--ckpt")
ap.add_argument("--probe-ckpt", help="standalone probe (train.py --probe-only); default <data>/probe_p<pool>.pt")
ap.add_argument("--stride", type=int, default=3, help="anchor every n ticks")
ap.add_argument("--out", default="results")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda"

ck = torch.load(args.ckpt or f"{args.data}/wm_p{args.pool}.pt", weights_only=False)
H, K, n_tok = ck["args"]["H"], ck["args"]["K"], ck["n_tok"]
model, probe = Predictor(n_tok, max_t=H + K).to(dev).eval(), Probe(n_tok).to(dev).eval()
model.load_state_dict(ck["model"])
# one probe for the floor, the gate and decoding rollouts: trained on true frames only (encoder is frozen)
probe.load_state_dict(torch.load(args.probe_ckpt or f"{args.data}/probe_p{args.pool}.pt", weights_only=False)["probe"])

meta = np.load(f"{args.data}/meta.npz")
emb = torch.from_numpy(np.load(f"{args.data}/emb_p{args.pool}.npy", mmap_mode="r"))
S, contact, act, off = meta["states"], meta["contact"], meta["action"], meta["offsets"]
test = np.flatnonzero(episode_split(meta["profiles"]) == 0)
anchors = np.concatenate([np.arange(off[e] + H - 1, off[e + 1] - K, args.stride) for e in test])
ks = np.arange(1, K + 1)
fut = anchors[:, None] + ks  # (A, K) frame index of each predicted tick
print(f"{len(anchors)} anchors from {len(test)} test episodes")


@torch.no_grad()
def learned():
    out = {k: [] for k in ["agent", "block", "angle", "l1", "cos", "floor_agent", "floor_block", "floor_angle", "perc"]}
    for i in range(0, len(anchors), 128):
        a = anchors[i:i + 128]
        z = norm(emb[torch.from_numpy(a[:, None] + np.arange(-H + 1, K + 1))].to(dev))  # context + true future
        u = torch.from_numpy(norm_xy(act[a[:, None] + np.arange(-H + 1, K)]).astype(np.float32)).to(dev)
        with amp():
            pred = rollout(model, z[:, :H], u, K).float()
        true = z[:, H:]
        ag, bl, th, _ = decode(probe(pred))
        fa, fb, ft, _ = decode(probe(true))  # probe on the true future frame = perception floor
        pa, pb, pt, _ = decode(probe(z[:, H - 2:H]))  # probe's reading of the two newest delayed frames
        for k, v in [("agent", ag), ("block", bl), ("angle", th), ("floor_agent", fa), ("floor_block", fb), ("floor_angle", ft)]:
            out[k].append(v.cpu().numpy())
        out["l1"].append((pred - true).abs().mean((-1, -2)).cpu().numpy())
        out["cos"].append((1 - torch.cosine_similarity(pred, true, -1).mean(-1)).cpu().numpy())
        out["perc"].append(torch.cat([pa, pb, pt[..., None]], -1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


L = learned()
true_now, true_prev = S[anchors], S[anchors - 1]
perc = L["perc"]  # (A, 2, 5): [agent xy, block xy, angle] for frames tau-1, tau
perc_now = np.concatenate([perc[:, 1], np.zeros((len(perc), 5))], 1)[:, [0, 1, 5, 6, 2, 3, 4, 7, 8, 9]]
perc_prev = np.concatenate([perc[:, 0], np.zeros((len(perc), 5))], 1)[:, [0, 1, 5, 6, 2, 3, 4, 7, 8, 9]]


def baselines(now, prev):
    """Each returns (agent (A,K,2), block (A,K,2), angle (A,K)) from what one could read off the delayed frames."""
    rep = lambda x: np.repeat(x[:, None], K, 1)
    hold = (rep(now[:, AGENT]), rep(now[:, BLOCK]), rep(now[:, ANGLE]))
    kk = ks[None, :, None]
    cv = (now[:, None, AGENT] + kk * (now[:, AGENT] - prev[:, AGENT])[:, None],
          now[:, None, BLOCK] + kk * (now[:, BLOCK] - prev[:, BLOCK])[:, None],
          now[:, None, ANGLE] + ks[None] * wrap(now[:, ANGLE] - prev[:, ANGLE])[:, None])
    vel = (now[:, AGENT] - prev[:, AGENT]) / (TICK_MS / 1000)
    kin = (agent_pd(now[:, AGENT], vel, act[anchors[:, None] + np.arange(K)]), hold[1], hold[2])
    return {"hold": hold, "const-vel": cv, "kinematic": kin}


methods = {"learned": (L["agent"], L["block"], L["angle"]), "probe floor": (L["floor_agent"], L["floor_block"], L["floor_angle"])}
methods |= {f"{m} (true)": v for m, v in baselines(true_now, true_prev).items()}
methods |= {f"{m} (probe)": v for m, v in baselines(perc_now, perc_prev).items()}

truth = S[fut]
in_contact = np.array([np.cumsum(contact[a:a + K + 1])[1:] > 0 for a in anchors])  # any contact in [tau, tau+k]
moving = (np.linalg.norm(truth[..., BLOCK] - S[anchors, None, BLOCK], axis=-1) >= 5) | \
         (np.degrees(np.abs(wrap(truth[..., ANGLE] - S[anchors, None, ANGLE]))) >= 3)  # block really moved in [tau, tau+k]
phases = {"all": np.ones_like(in_contact), "contact": in_contact, "moving": moving, "free": ~in_contact}
rows = []
for name, (ag, bl, th) in methods.items():
    err = {"agent_px": np.linalg.norm(ag - truth[..., AGENT], axis=-1), "block_px": np.linalg.norm(bl - truth[..., BLOCK], axis=-1),
           "angle_deg": np.degrees(np.abs(wrap(th - truth[..., ANGLE])))}
    if name == "learned":
        err |= {"latent_l1": L["l1"], "latent_cos": L["cos"]}
    for ph, m in phases.items():
        for j, k in enumerate(ks):
            sel = m[:, j]
            rows.append({"method": name, "phase": ph, "k": k, "ms": k * TICK_MS, "n": int(sel.sum()),
                         **{e: float(v[sel, j].mean()) if sel.any() else float("nan") for e, v in err.items()}})

with open(f"{args.out}/g1_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["latent_l1", "latent_cos"], restval="")
    w.writeheader()
    w.writerows(rows)

# --- G1 verdict: the pre-registered criterion in README.md ----------------------------------------------------
get = lambda m, ph, k, e: next(r[e] for r in rows if r["method"] == m and r["phase"] == ph and r["k"] == k)
base = ["hold (true)", "const-vel (true)", "kinematic (true)"]
verdict = {"floor_block_px": get("probe floor", "all", 1, "block_px"), "floor_angle_deg": get("probe floor", "all", 1, "angle_deg")}
ok = verdict["floor_block_px"] < 3 and verdict["floor_angle_deg"] < 3
for k in (10, 20):
    for e in ("block_px", "angle_deg"):
        best = min(get(b, "moving", k, e) for b in base)
        mine = get("learned", "moving", k, e)
        verdict[f"moving_{k * TICK_MS}ms_{e}"] = {"learned": mine, "best_baseline": best, "ratio": mine / best}
        ok &= mine <= 0.8 * best
verdict["G1"] = "PASS" if ok else "FAIL"
json.dump(verdict, open(f"{args.out}/g1_verdict.json", "w"), indent=2)
print(json.dumps(verdict, indent=2))

# --- chart -----------------------------------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = {"learned": "#2a78d6", "hold (true)": "#eb6834", "const-vel (true)": "#1baf7a", "kinematic (true)": "#eda100"}
labels = {"kinematic (true)": "kinematic (true; block = hold)"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "font.size": 9})
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), facecolor="#fcfcfb", sharex=True)
for r, ph in enumerate(["moving", "free"]):
    for c, (e, label) in enumerate([("block_px", "block position error (px)"), ("angle_deg", "block angle error (deg)"),
                                    ("agent_px", "agent position error (px)")]):
        ax = axes[r, c]
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for m, col in colors.items():
            ax.plot(ks * TICK_MS, [get(m, ph, k, e) for k in ks], color=col, lw=2.5 if m == "learned" else 2, label=labels.get(m, m))
        ax.plot(ks * TICK_MS, [get("probe floor", ph, k, e) for k in ks], color=colors["learned"], lw=1.2, ls="--",
                label="probe floor (perception only)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{label} — {'block moving' if ph == 'moving' else 'no contact'}", fontsize=9, color="#0b0b0b", loc="left")
        if r == 1:
            ax.set_xlabel("prediction horizon (ms)")
h, lab = axes[0, 0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", ncol=5, frameon=False)
fig.suptitle(f"G1: {verdict['G1']}", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(f"{args.out}/g1.png", dpi=150)
print(f"wrote {args.out}/g1.png, g1_metrics.csv, g1_verdict.json")
