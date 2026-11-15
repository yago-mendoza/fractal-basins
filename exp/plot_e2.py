"""Plot an E2 zoom sequence: one panel per level, each with the next level's window outlined."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

d = np.load(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(".npz", ".png")
n = 1 + max(int(k[5:].split("_")[0]) for k in d.files if k.startswith("level"))
fig, axs = plt.subplots(1, n, figsize=(4.0 * n, 4.3))
for lv, ax in enumerate(axs):
    s = d[f"level{lv}_settle"]
    cx, cy, w = d[f"level{lv}_box"]
    ext = (cx - w / 2, cx + w / 2, cy - w / 2, cy + w / 2)
    ax.imshow(s, origin="lower", cmap="magma", interpolation="nearest", extent=ext)
    title = f"level {lv}: width {w:g}"
    if f"level{lv}_metrics" in d.files:
        title += f"\nSb={d[f'level{lv}_metrics'][0]:.2f}  alpha={d[f'level{lv}_metrics'][2]:.2f}"
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if lv + 1 < n:
        nx, ny, nw = d[f"level{lv + 1}_box"]
        ax.add_patch(Rectangle((nx - nw / 2, ny - nw / 2), nw, nw, fill=False, ec="cyan", lw=1.2))
fig.tight_layout()
fig.savefig(out, dpi=110)
print("wrote", out)
