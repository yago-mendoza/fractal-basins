"""Export what the MOVIOLA (HyperFrames) video needs from the V1 runs: the map and its zoom levels as PNGs, and
the numbers (pair, deep pair, Sudoku run, colour scale) as JSON. Same data and colours as exp/video.py.

Images are oriented for the screen (row 0 at the top = largest y). Coordinates in the JSON are plane coordinates
(x right, y up), each image with the plane rectangle it covers (its pixels are centred on its grid, so it reaches
half a pixel past the grid on every side).

usage: python exp/moviola_export.py out/v1_small.npz out/v1_zoom.npz OUT_DIR
"""
import json
import os
import sys

import numpy as np
import matplotlib
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SUDOKUS  # noqa: E402

SMALL, ZOOMF, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
MAGMA, GAMMA = matplotlib.colormaps["magma"], 0.55

d = np.load(SMALL)
SETTLE, SOL = d["settle"], d["sol"]
RES = SETTLE.shape[0]
VMAX = int(SETTLE.max())
ROW, CA, CB = (int(x) for x in d["pair"])
z = np.load(ZOOMF)
LEVELS = [(SETTLE, np.array([0.0, 0.0, 2.0]))] + [(z[f"level{k}_settle"], z[f"level{k}_box"])
                                                   for k in range(1, 9) if f"level{k}_box" in z.files]


def rgb(s):
    return (MAGMA(np.clip(np.asarray(s, float) / VMAX, 0, 1) ** GAMMA)[..., :3] * 255).astype(np.uint8)


levels = []
for k, (f, b) in enumerate(LEVELS):
    n = f.shape[0]
    h = b[2] / (n - 1)
    Image.fromarray(rgb(f)[::-1]).save(os.path.join(OUT, f"nivel{k}.png"))
    levels.append({"img": f"nivel{k}.png", "n": n, "h": h, "cx": float(b[0]), "cy": float(b[1]),
                   "size": float(b[2] + h), "zoom": float(2.0 / b[2])})

G = np.linspace(-1, 1, RES)
pair = [{"x": float(G[CA]), "y": float(G[ROW]), "loops": int(SETTLE[ROW, CA])},
        {"x": float(G[CB]), "y": float(G[ROW]), "loops": int(SETTLE[ROW, CB])}]
za, zb = d["z_a"].astype(float), d["z_b"].astype(float)
rel = float(np.linalg.norm(za - zb) / np.linalg.norm(zb))

# two side-by-side pixels at the deepest zoom that disagree the most, near the middle (as in video.py)
f, b = LEVELS[-1]
n = f.shape[0]
diff = np.abs(f[:, 1:].astype(int) - f[:, :-1].astype(int))
m = np.zeros_like(diff, bool)
m[n // 4:3 * n // 4, n // 4:3 * n // 4] = True
rr, cc = np.unravel_index(np.argmax(np.where(m, diff, -1)), diff.shape)
gx = np.linspace(b[0] - b[2] / 2, b[0] + b[2] / 2, n)
gy = np.linspace(b[1] - b[2] / 2, b[1] + b[2] / 2, n)
deep = [{"x": float(gx[cc + i]), "y": float(gy[rr]), "loops": int(f[rr, cc + i])} for i in (0, 1)]

slow = d["trace_b"]                      # the slow start of the pair (column CB)
s_slow = int(SETTLE[ROW, CB])
run = [[int(v) for v in slow[k]] for k in range(s_slow + 1)]

out = {
    "vmax": VMAX, "gamma": GAMMA,
    "scale": ["#%02x%02x%02x" % tuple(rgb([v])[0]) for v in range(1, VMAX + 1)],
    "levels": levels, "pair": pair, "pair_rel_diff": rel, "deep_pair": deep,
    "sudoku": {"puzzle": SUDOKUS["medium"], "solution": [int(v) for v in SOL], "run": run, "settles_at": s_slow},
}
json.dump(out, open(os.path.join(OUT, "datos.json"), "w"), indent=1)
print(f"levels {len(levels)}, pair {pair}, rel {rel:.4f}, deep {deep}, run {len(run)} loops")
