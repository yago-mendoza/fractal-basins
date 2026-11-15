"""E5. Does the model's chaos track the problem's own difficulty?

Sudoku-Extreme (sapientinc/sudoku-extreme, test split) rates every puzzle by the number of
backtracks the tdoku solver needs, a measure of difficulty with no neural network in it.
Pick puzzles across log-spaced rating bins and, for each, draw a small EqR basin map on the
E1 plane (seed 0) and score it.

usage: python e5_difficulty.py CSV RES MAX_STEPS PER_BIN OUT
"""
import csv
import sys
import time

import numpy as np
import torch

from loopscape import get_solver
from loopscape.metrics import basin_entropy, uncertainty_exponent
from loopscape.utils import batched_apply

from common import DEV, mesh, plane, to_traces, settle_times

CSV, RES, MAX_STEPS, PER_BIN, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
BINS = [(0, 0), (1, 2), (3, 6), (7, 14), (15, 30), (31, 62), (63, 126), (127, 254), (255, 10**6)]

rows = list(csv.DictReader(open(CSV)))
rng = np.random.default_rng(0)
chosen = []
for lo, hi in BINS:
    cand = [r for r in rows if lo <= int(r["rating"]) <= hi]
    for i in rng.choice(len(cand), PER_BIN, replace=False):
        chosen.append(cand[i])
print(len(chosen), "puzzles", flush=True)

solver = get_solver("eqr", device=DEV)
u, v = plane((97, 512), seed=0)
pts = mesh(RES)
rec = {k: [] for k in ["rating", "correct", "unsettled", "settle_mean", "settle_median",
                       "sb", "sbb", "alpha", "n_finals"]}
fields = []
for j, r in enumerate(chosen):
    t0 = time.time()
    q, a = r["question"], r["answer"]

    def f(p):
        z = p[:, 0, None, None] * u + p[:, 1, None, None] * v
        out = solver.solve_batch(q, z_H=z, z_L=z * 0, max_steps=MAX_STEPS, noise_scale=0.0,
                                 return_intermediates=True)
        return to_traces(out)

    tr = batched_apply(f, pts, batch_size=512, pad_dim=1)
    settle = settle_times(tr).reshape(RES, RES)
    final = tr[:, -1].reshape(RES * RES, 81).numpy()
    sol = np.array([int(c) for c in a], dtype=final.dtype)
    ok = (final == sol).all(1)
    sb = basin_entropy(settle, box_size=4)
    ue = uncertainty_exponent(settle, box_size=4)
    vals = dict(rating=int(r["rating"]), correct=ok.mean(), unsettled=(settle >= MAX_STEPS - 1).mean(),
                settle_mean=settle.mean(), settle_median=np.median(settle), sb=sb["basin_entropy"],
                sbb=sb["boundary_basin_entropy"], alpha=ue["uncertainty_exponent"],
                n_finals=len(np.unique(final, axis=0)))
    for k, x in vals.items():
        rec[k].append(x)
    fields.append(settle)
    print(f"{j:2d} rating {vals['rating']:4d} {time.time() - t0:5.1f}s correct {vals['correct']:.2f} "
          f"unsettled {vals['unsettled']:.2f} settle mean {vals['settle_mean']:5.1f} "
          f"Sb {vals['sb']:.2f} alpha {vals['alpha']:.2f} finals {vals['n_finals']}", flush=True)
    np.savez_compressed(OUT, fields=np.array(fields), questions=np.array([c["question"] for c in chosen[:j + 1]]),
                        **{k: np.array(x) for k, x in rec.items()})

r_ = np.array(rec["rating"], float)
from scipy.stats import spearmanr
for k in ["settle_mean", "unsettled", "sb", "alpha", "correct"]:
    rho, p = spearmanr(r_, rec[k])
    print(f"Spearman rating vs {k}: rho {rho:+.2f} (p {p:.1e})")
print("saved", OUT)
