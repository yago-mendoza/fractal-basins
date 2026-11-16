"""V1, part 2: pick two neighbouring starts of the V1 map, one fast and one slow, and pack what the
video needs into a small file (the full traces stay on the GPU machine).

A pair is two horizontally adjacent pixels, both ending on the solution, the first settled by
loop 2 and the second at loop 12 or later, away from the edges; among those, the one whose
surroundings mix fast and slow the most.

usage: python v1_pick.py V1_NPZ OUT_NPZ
"""
import sys

import numpy as np
from scipy.ndimage import uniform_filter

from common import mesh, plane

d = np.load(sys.argv[1])
settle, wrong, changed, sol, traces = d["settle"], d["wrong"], d["changed"], d["sol"], d["traces"]
RES = settle.shape[0]
ok = (traces[:, -1] == sol).all(-1).reshape(RES, RES)
s = settle.astype(float)
mix = uniform_filter(s ** 2, 7) - uniform_filter(s, 7) ** 2

best = None
for r in range(20, RES - 20):
    for c in range(20, RES - 21):
        for ca, cb in ((c, c + 1), (c + 1, c)):
            if ok[r, ca] and ok[r, cb] and settle[r, ca] <= 2 and settle[r, cb] >= 12:
                score = mix[r, c] + 2 * settle[r, cb]
                if best is None or score > best[0]:
                    best = (score, r, ca, cb)
_, r, ca, cb = best
ia, ib = r * RES + ca, r * RES + cb
print(f"pair at row {r}: col {ca} settles at {settle[r, ca]}, col {cb} at {settle[r, cb]}")

pts = mesh(RES).numpy()
u, v = plane((97, 512), seed=0)
z_a = (pts[ia, 0] * u + pts[ia, 1] * v).numpy().astype(np.float16)
z_b = (pts[ib, 0] * u + pts[ib, 1] * v).numpy().astype(np.float16)
np.savez_compressed(sys.argv[2], settle=settle, wrong=wrong, changed=changed, sol=sol,
                    pair=np.array([r, ca, cb]), pts=pts[[ia, ib]], trace_a=traces[ia],
                    trace_b=traces[ib], z_a=z_a, z_b=z_b)
print("saved", sys.argv[2])
