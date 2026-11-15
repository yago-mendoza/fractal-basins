"""E4. What a slow start does while it is slow (EqR, medium Sudoku, the E1 plane).

Pick starts from the E1 map: `fast` ones in the smooth region (settle <= 2, whole 5x5
neighbourhood too) and `slow` ones in the speckle (settle >= 12). Run each for 24 loops and
record, loop by loop:
  - wrong cells: cells of the decoded grid that differ from the solution,
  - step size: RMS change of the latent state z_H from one loop to the next.
If slow starts linger near a saddle, they should sit for several loops at a few wrong cells
with a small step size before dropping to zero.

usage: python e4_saddles.py E1_NPZ N_PER_GROUP OUT
"""
import sys

import numpy as np
import torch

from loopscape import get_solver

from common import DEV, SUDOKUS, mesh, plane, to_traces

E1, N, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
S = np.load(E1)["medium_settle"]
res = S.shape[0]
pts = mesh(res)                                   # index = iy * res + ix, same as E1
rng = np.random.default_rng(0)

pad = np.pad(S, 2, mode="edge")
nbmax = np.max([pad[dy:dy + res, dx:dx + res] for dy in range(5) for dx in range(5)], axis=0)
fast = np.flatnonzero((nbmax <= 2).ravel())
slow = np.flatnonzero((S >= 12).ravel())
print("candidates fast", len(fast), "slow", len(slow))
pick = {"fast": rng.choice(fast, min(N, len(fast)), replace=False),
        "slow": rng.choice(slow, min(N, len(slow)), replace=False)}

solver = get_solver("eqr", device=DEV)
u, v = plane((97, 512), seed=0)
out = {}
for name, idx in pick.items():
    p = pts[torch.as_tensor(idx)]
    z = p[:, 0, None, None] * u + p[:, 1, None, None] * v
    r = solver.solve_batch(SUDOKUS["medium"], z_H=z, z_L=z * 0, max_steps=24, noise_scale=0.0,
                           return_intermediates=True, return_latents=True)
    tr = to_traces(r).numpy().reshape(len(idx), -1, 81)          # (B, steps, 81)
    final = tr[:, -1]
    assert all(r.solved), "medium was always solved in E1"
    wrong = (tr != final[:, None]).sum(-1)                         # (B, steps)
    step = np.asarray(r.residual_history, dtype=float).T           # (B, steps)
    out[f"{name}_idx"] = idx
    out[f"{name}_settle"] = S.ravel()[idx]
    out[f"{name}_wrong"] = wrong
    out[f"{name}_step"] = step
    med_w = np.median(wrong, 0)
    print(name, "median wrong cells per loop:", med_w.astype(int).tolist())
    print(name, "median step per loop:", np.round(np.median(step, 0), 4).tolist())
    # loops spent at 1..3 wrong cells before reaching 0
    near = ((wrong >= 1) & (wrong <= 3)).sum(1)
    print(name, f"loops spent 1-3 cells from the answer: median {np.median(near):.0f}, "
          f"mean {near.mean():.1f}, max {near.max()}", flush=True)
np.savez_compressed(OUT, **out)
print("saved", OUT)
