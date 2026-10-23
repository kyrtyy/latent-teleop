"""C2 gate (frozen in README): on 300j and 300b, learned - none > 0 and learned - kinematic > 0 in coverage AUC,
each with a paired-bootstrap 95% CI excluding 0. Also prints gap recovered = (ghost - none) / (0ms - none)."""
import os
import sys

import numpy as np


def auc(d, profile, kind):
    f = f"{d}/{profile}_{kind}.npz"
    return np.load(f)["coverage"].mean(1) if os.path.exists(f) else None


def ci(x, n=10000, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "results/c2"
    ref = auc(d, "0", "none").mean()
    gate = []
    for p in ["50j", "150j", "150b", "300j", "300b"]:
        a = {k: auc(d, p, k) for k in ["none", "kinematic", "learned", "oracle"]}
        row = "  ".join(f"{k} {v.mean():.3f} ({(v.mean() - a['none'].mean()) / (ref - a['none'].mean()):4.0%})"
                        for k, v in a.items() if v is not None)
        print(f"{p:>5}: {row}")
        if a["learned"] is None:
            continue
        for base in ["none", "kinematic"]:
            diff = a["learned"] - a[base]
            lo, hi = ci(diff)
            print(f"        learned - {base:<9} {diff.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
            if p in ("300j", "300b"):
                gate.append(lo > 0)
    print(f"0 ms reference {ref:.3f}")
    print("C2:", "PASS" if len(gate) == 4 and all(gate) else "pending" if len(gate) < 4 else "FAIL")
