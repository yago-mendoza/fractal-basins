"""E6, part 2: replay start-and-stop policies on the recorded traces of e6_starts.py.

Every policy only sees what a deployed system would see: the decoded grid after each loop.
An answer is *accepted* once it has not changed for k consecutive loops. Correctness is
checked against the dataset's solution afterwards, never used to decide.

Policies, each given a total budget of B loops per puzzle:
  wait        one random start, run until accepted or until the budget (at most MAX) runs out
  restart(T)  random start; if not accepted by loop T, drop it and start a new random one
  evolved     the CMA-ES start first, then random restarts with the same T
  zero        the all-zeros start first, then random restarts with the same T
If the budget runs out, the last decoded grid is the answer.

usage: python e6_policies.py E6_NPZ [OUT_JSON]
"""
import json
import sys

import numpy as np

d = np.load(sys.argv[1])
TR, SOL = d["traces"], d["sols"]                   # (P, K+2, MAX, 81), (P, 81)
P, NS, MAX, _ = TR.shape
K = NS - 2
EVO, ZERO = K, K + 1


def accept_time(tr, k):
    """First loop index t (0-based) at which the grid has equalled the previous k-1 ones."""
    same = (tr[1:] == tr[:-1]).all(-1)             # same[t-1]: grid t equals grid t-1
    run = 0
    for t, s in enumerate(same, start=1):
        run = run + 1 if s else 0
        if run >= k - 1:
            return t
    return None


# precompute per (puzzle, start): accept time and correctness of the accepted / final grid
def table(k):
    acc_t = np.full((P, NS), -1)
    acc_ok = np.zeros((P, NS), bool)
    grid_ok = np.zeros((P, NS, MAX), bool)
    for p in range(P):
        grid_ok[p] = (TR[p] == SOL[p]).all(-1)
        for s in range(NS):
            t = accept_time(TR[p, s], k)
            if t is not None:
                acc_t[p, s], acc_ok[p, s] = t, grid_ok[p, s, t]
    return acc_t, acc_ok, grid_ok


def simulate(order, B, T, acc_t, acc_ok, grid_ok, p):
    """Returns (right, loops used). T=None means never restart (one start, at most MAX loops)."""
    used = 0
    for s in order:
        cap = min(MAX if T is None else T, B - used, MAX)
        t = acc_t[p, s]
        if 0 <= t < cap:
            return acc_ok[p, s], used + t + 1
        used += cap
        if T is None or used >= B:
            return grid_ok[p, s, cap - 1], used
    return False, used


rng = np.random.default_rng(0)
results = {}
for k in (3, 5):
    acc_t, acc_ok, grid_ok = table(k)
    acc_rand = acc_t[:, :K]
    print(f"\nk={k}: random starts accepted within {MAX} loops: {(acc_rand >= 0).mean():.2f}; "
          f"accepted and right: {acc_ok[:, :K][acc_rand >= 0].mean():.3f}; "
          f"evolved accepted {(acc_t[:, EVO] >= 0).mean():.2f}, zero accepted {(acc_t[:, ZERO] >= 0).mean():.2f}")
    for B in (24, 32, 48, 64, 96, 128):
        row = {}
        for name, T, first in [("wait", None, None), ("restart20", 20, None), ("restart30", 30, None),
                               ("evolved+restart30", 30, EVO), ("zero+restart30", 30, ZERO)]:
            oks, loops = [], []
            for p in range(P):
                for rep in range(64):
                    order = list(rng.permutation(K))
                    if first is not None:
                        order = [first] + order
                    ok, used = simulate(order, B, T, acc_t, acc_ok, grid_ok, p)
                    oks.append(ok); loops.append(used)
            row[name] = (float(np.mean(oks)), float(np.mean(loops)))
        results[f"k{k}_B{B}"] = row
        print(f"B={B:3d} " + "  ".join(f"{n}: {a:.2f} ({l:4.1f} loops)" for n, (a, l) in row.items()))

# percentile of the evolved start among random starts, per held-out puzzle (by settle loop, k=3)
acc_t, _, _ = table(3)
pct = []
for p in range(P):
    r = np.where(acc_t[p, :K] >= 0, acc_t[p, :K], 10**6)
    e = acc_t[p, EVO] if acc_t[p, EVO] >= 0 else 10**6
    pct.append(float((r > e).mean() + 0.5 * (r == e).mean()))
print("\nevolved start, fraction of random starts it beats on each held-out puzzle:",
      np.round(pct, 2).tolist(), " median", np.median(pct))
results["evolved_percentile"] = pct
if len(sys.argv) > 2:
    json.dump(results, open(sys.argv[2], "w"), indent=1)
