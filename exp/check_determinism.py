"""Is the speckle real? Same 256 starting points, run (a) twice in the same batch,
(b) shuffled into a different batch order, (c) mixed with 256 other points.
If settle times differ between runs, the speckle is (at least partly) numerical noise."""
import numpy as np
import torch
from loopscape import get_solver
from common import DEV, SUDOKUS, mesh, plane, to_traces, settle_times

s = get_solver("eqr", device=DEV)
u, v = plane((97, 512), seed=0)
p = mesh(16, center=(-0.30153, 0.58124), width=0.03125)  # the deepest medium zoom
other = mesh(16, center=(0.5, -0.5), width=0.5)


def run(pts):
    z = pts[:, 0, None, None] * u + pts[:, 1, None, None] * v
    r = s.solve_batch(SUDOKUS["medium"], z_H=z, z_L=z * 0, max_steps=24,
                      noise_scale=0.0, return_intermediates=True)
    return settle_times(to_traces(r))


a = run(p)
b = run(p)
perm = torch.randperm(len(p))
c = np.empty_like(a); c[perm.numpy()] = run(p[perm])
d = run(torch.cat([other, p]))[len(other):]
print("settle range", a.min(), a.max())
print("same batch, rerun:   differ", (a != b).sum(), "/", len(a))
print("shuffled order:      differ", (a != c).sum(), "/", len(a))
print("mixed with others:   differ", (a != d).sum(), "/", len(a))
