"""E1. EqR basin slices on three Sudokus of rising difficulty.

For each puzzle: take a random 2D plane through the (97, 512) latent space, start the
loop from every point of a res x res grid on that plane, and record
  - settle time: loops until the decoded grid stops changing for good,
  - solved: whether the final grid is a valid solution,
  - the final grid itself (to tell distinct attractors apart).
Then score each settle-time field with loopscape's basin entropy and uncertainty exponent.

usage: python e1_sudoku_slices.py RES [MAX_STEPS] [OUT] [easy,medium,hard]
"""
import sys
import time

import numpy as np
import torch

from loopscape import get_solver
from loopscape.metrics import basin_entropy, uncertainty_exponent
from loopscape.utils import batched_apply

from common import DEV, SUDOKUS, mesh, plane, to_traces, settle_times

RES = int(sys.argv[1]) if len(sys.argv) > 1 else 50
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
OUT = sys.argv[3] if len(sys.argv) > 3 else f"e1_res{RES}.npz"
NAMES = sys.argv[4].split(",") if len(sys.argv) > 4 else list(SUDOKUS)

solver = get_solver("eqr", device=DEV)
u, v = plane((97, 512), seed=0)
pts = mesh(RES)
out = {}

for name in NAMES:
    puzzle = SUDOKUS[name]
    t0 = time.time()
    finals, solved = [], []

    def f(p):
        z = p[:, 0, None, None] * u + p[:, 1, None, None] * v
        r = solver.solve_batch(puzzle, z_H=z, z_L=z * 0, max_steps=MAX_STEPS,
                               noise_scale=0.0, return_intermediates=True)
        solved.append(torch.tensor(np.asarray(r.solved)))
        return to_traces(r)

    traces = batched_apply(f, pts, batch_size=512, pad_dim=1)
    settle = settle_times(traces).reshape(RES, RES)
    final = traces[:, -1].reshape(RES * RES, 81).numpy()
    ok = torch.cat(solved).numpy()[: RES * RES].reshape(RES, RES)
    # label each distinct final grid (attractor id) by first appearance
    _, attractor = np.unique(final, axis=0, return_inverse=True)
    attractor = attractor.reshape(RES, RES)

    sb = basin_entropy(settle, box_size=5)
    ue = uncertainty_exponent(settle, box_size=5)
    dt = time.time() - t0
    print(f"{name:6s} res={RES} {dt:6.1f}s  solved={ok.mean():.3f}  "
          f"distinct finals={attractor.max() + 1}  settle min/med/max="
          f"{settle.min()}/{int(np.median(settle))}/{settle.max()}  "
          f"Sb={sb['basin_entropy']:.3f} Sbb={sb.get('boundary_basin_entropy', float('nan')):.3f}  "
          f"alpha={ue['uncertainty_exponent']:.3f}", flush=True)
    out[f"{name}_settle"] = settle
    out[f"{name}_solved"] = ok
    out[f"{name}_attractor"] = attractor
    out[f"{name}_metrics"] = np.array([sb["basin_entropy"], sb.get("boundary_basin_entropy", np.nan),
                                       ue["uncertainty_exponent"]])

np.savez_compressed(OUT, **out)
print("saved", OUT)
