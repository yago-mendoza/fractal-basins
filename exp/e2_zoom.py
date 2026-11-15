"""E2. Successive zooms into a basin boundary (EqR, same plane as E1).

Start from an E1 field; each level re-centres on the most mixed spot of the previous
image (highest settle-time variance in a 9x9 window, away from the edges) and shrinks the
window by ZOOM. If the boundary is fractal, structure keeps showing up at every level and
the uncertainty exponent stays roughly the same.

usage: python e2_zoom.py PUZZLE START_NPZ RES MAX_STEPS LEVELS ZOOM OUT
"""
import sys
import time

import numpy as np
import torch
from scipy.ndimage import uniform_filter

from loopscape import get_solver
from loopscape.metrics import basin_entropy, uncertainty_exponent
from loopscape.utils import batched_apply

from common import DEV, SUDOKUS, mesh, plane, to_traces, settle_times

name, start, RES, MAX_STEPS, LEVELS, ZOOM, OUT = sys.argv[1:8]
RES, MAX_STEPS, LEVELS, ZOOM = int(RES), int(MAX_STEPS), int(LEVELS), float(ZOOM)
puzzle = SUDOKUS[name]

solver = get_solver("eqr", device=DEV)
u, v = plane((97, 512), seed=0)


def most_mixed(settle, center, width):
    """(a, b) of the pixel with the highest local settle-time variance."""
    s = settle.astype(float)
    var = uniform_filter(s ** 2, 9) - uniform_filter(s, 9) ** 2
    n = s.shape[0]
    m = n // 8
    var[:m] = var[-m:] = 0
    var[:, :m] = var[:, -m:] = 0
    iy, ix = np.unravel_index(np.argmax(var), var.shape)
    g = np.linspace(-width / 2, width / 2, n)
    return (center[0] + g[ix], center[1] + g[iy])


prev = np.load(start)[f"{name}_settle"]
center, width = (0.0, 0.0), 2.0
out = {"level0_settle": prev, "level0_box": np.array([*center, width])}

for lv in range(1, LEVELS + 1):
    center = most_mixed(prev, center, width)
    width = width / ZOOM
    t0 = time.time()

    def f(p):
        z = p[:, 0, None, None] * u + p[:, 1, None, None] * v
        r = solver.solve_batch(puzzle, z_H=z, z_L=z * 0, max_steps=MAX_STEPS,
                               noise_scale=0.0, return_intermediates=True)
        return to_traces(r)

    traces = batched_apply(f, mesh(RES, center, width), batch_size=512, pad_dim=1)
    settle = settle_times(traces).reshape(RES, RES)
    sb = basin_entropy(settle, box_size=5)
    ue = uncertainty_exponent(settle, box_size=5)
    print(f"{name} level {lv}: center=({center[0]:+.5f},{center[1]:+.5f}) width={width:.5f} "
          f"{time.time() - t0:6.1f}s settle {settle.min()}..{settle.max()} "
          f"Sb={sb['basin_entropy']:.3f} Sbb={sb['boundary_basin_entropy']:.3f} "
          f"alpha={ue['uncertainty_exponent']:.3f}", flush=True)
    out[f"level{lv}_settle"] = settle
    out[f"level{lv}_box"] = np.array([*center, width])
    out[f"level{lv}_metrics"] = np.array([sb["basin_entropy"], sb["boundary_basin_entropy"],
                                          ue["uncertainty_exponent"]])
    np.savez_compressed(OUT, **out)
    prev = settle

print("saved", OUT)
