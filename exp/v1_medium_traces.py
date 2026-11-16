"""V1. The medium Sudoku map of E1 again, keeping every decoded grid, for the video.

Same plane (seed 0), same grid of starts, same loop budget as E1. Besides the settle-time map
it keeps, for every start and loop, the decoded grid (int8), how many cells differ from the
solution and whether the grid changed since the previous loop.

usage: python v1_medium_traces.py RES MAX_STEPS OUT_NPZ
"""
import sys
import time

import numpy as np
import torch

from loopscape import get_solver
from loopscape.utils import batched_apply

from common import DEV, SUDOKUS, mesh, plane, to_traces, settle_times

RES, MAX_STEPS, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
puzzle = SUDOKUS["medium"]
solver = get_solver("eqr", device=DEV)
u, v = plane((97, 512), seed=0)
t0 = time.time()


def f(p):
    z = p[:, 0, None, None] * u + p[:, 1, None, None] * v
    r = solver.solve_batch(puzzle, z_H=z, z_L=z * 0, max_steps=MAX_STEPS,
                           noise_scale=0.0, return_intermediates=True)
    print(f"  batch of {len(p)} at {time.time() - t0:6.1f}s", flush=True)
    return to_traces(r)


traces = batched_apply(f, mesh(RES), batch_size=512, pad_dim=1)          # (N, steps, 9, 9)
settle = settle_times(traces).reshape(RES, RES)
flat = traces.reshape(len(traces), MAX_STEPS, 81).numpy().astype(np.int8)
finals, counts = np.unique(flat[:, -1], axis=0, return_counts=True)
sol = finals[np.argmax(counts)]                                          # the grid most starts end on
wrong = (flat != sol).sum(-1).astype(np.uint8)                           # (N, steps)
changed = np.concatenate([np.ones((len(flat), 1), bool), (flat[:, 1:] != flat[:, :-1]).any(-1)], 1)
print(f"done {time.time() - t0:.0f}s: settle {settle.min()}..{settle.max()}, "
      f"{len(finals)} distinct finals, {np.mean((flat[:, -1] == sol).all(-1)):.4f} end on the main one",
      flush=True)
np.savez_compressed(OUT, settle=settle, wrong=wrong, changed=changed, sol=sol, traces=flat)
print("saved", OUT, flush=True)
