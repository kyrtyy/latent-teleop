"""C3 and RQ4 gate verdicts (criteria frozen in README). Paired bootstrap from c2_gate."""
import glob
import os
import sys

import numpy as np

from c2_gate import ci


def c3(d="results/c3", safe="results/c3safe"):
    L = lambda p, a, dd=d: np.load(f"{dd}/{p}_{a}.npz")
    ok = []
    for p in ["50j", "150j", "150b", "300j", "300b"]:
        s, b, c = L(p, "setpoint"), L(p, "blend"), L(p, "chunk")
        dp, dc = b["pauses_per_min"] - s["pauses_per_min"], b["coverage"].mean(1) - s["coverage"].mean(1)
        (plo, phi), (clo, chi) = ci(dp), ci(dc)
        print(f"{p:>5}: pauses/min {s['pauses_per_min'].mean():6.1f} -> {b['pauses_per_min'].mean():5.1f} [{plo:+.1f}, {phi:+.1f}]  "
              f"AUC {s['coverage'].mean():.3f} -> {b['coverage'].mean():.3f} (chunk {c['coverage'].mean():.3f}) "
              f"diff [{clo:+.4f}, {chi:+.4f}]  jerk {s['jerk'].mean():.3f} -> {b['jerk'].mean():.3f}  "
              f"success {s['success'].mean():.2f} -> {b['success'].mean():.2f}")
        if p in ("300j", "300b"):
            ok += [phi < 0, clo > -0.01]
    print("C3:", "PASS" if all(ok) else "FAIL")
    for p in ["300j", "300b"]:
        for a in ["oracle", "wm"]:
            f = f"{safe}/{p}_{a}.npz"
            if not os.path.exists(f):
                continue
            x, b = np.load(f), L(p, "blend")
            n = len(x["seeds"])
            dpush, dcov = x["robot_push"] - b["robot_push"][:n], x["coverage"].mean(1) - b["coverage"][:n].mean(1)
            print(f"  {p} {a:>6} vs blend (N={n}): robot push {b['robot_push'][:n].mean():6.1f} -> {x['robot_push'].mean():6.1f}px "
                  f"{ci(dpush)}  AUC diff {dcov.mean():+.4f} {ci(dcov)}  stops/min {x['stops_per_min'].mean():.0f}")


def rq4(d="results/rq4"):
    arms = sorted({os.path.basename(f).rsplit("_s", 1)[0] for f in glob.glob(f"{d}/*_s*.npz")})
    auc = {}
    for a in arms:  # per eval episode, averaged over training seeds
        runs = [np.load(f) for f in sorted(glob.glob(f"{d}/{a}_s*.npz"))]
        auc[a] = np.mean([r["coverage"].mean(1) for r in runs], 0)
        print(f"{a:>16}: {len(runs)} seeds, AUC {auc[a].mean():.4f} (seed means {[round(r['coverage'].mean(), 4) for r in runs]}) "
              f"success {np.mean([r['success'].mean() for r in runs]):.3f}  dropped {int(runs[0]['dropped'])}/{int(runs[0]['segments'])}")
    for x, y in [("oracle", "random_oracle"), ("oracle", "full"), ("wm", "random_wm"), ("wm", "full"), ("rules", "random_rules"),
                 ("rules", "full")] + [(f"oracle_{k}", "full") for k in ["hesitate", "wrong", "spike", "slip"]]:
        if x in auc and y in auc:
            dd = auc[x] - auc[y]
            print(f"  {x} - {y}: {dd.mean():+.4f} 95% CI {np.round(ci(dd), 4)}")
    if "wm" in auc and "random_wm" in auc and "full" in auc:
        lo = ci(auc["wm"] - auc["random_wm"])[0]
        print("RQ4:", "PASS" if lo > 0 and (auc["wm"] - auc["full"]).mean() >= -0.01 else "FAIL")


if __name__ == "__main__":
    {"c3": c3, "rq4": rq4}[sys.argv[1]]()
