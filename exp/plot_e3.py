"""Plot E3: settle-time fields per checkpoint (log colour), released model last."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

d = np.load(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(".npz", ".png")
tags = [k[:-7] for k in d.files if k.endswith("_settle")]
vmax = max(d[t + "_settle"].max() for t in tags)
fig, axs = plt.subplots(1, len(tags), figsize=(3.0 * len(tags), 3.6), squeeze=False)
for ax, t in zip(axs[0], tags):
    acc, sb, sbb, a = d[t + "_metrics"]
    im = ax.imshow(np.clip(d[t + "_settle"], 1, None), origin="lower", cmap="magma",
                   norm=LogNorm(1, vmax), interpolation="nearest")
    label = "released" if t == "released" else t.replace("step", "step ")
    ax.set_title(f"{label}\ncorrect {acc:.0%}  alpha={a:.2f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axs[0].tolist(), fraction=0.02, label="loops to settle")
fig.savefig(out, dpi=110, bbox_inches="tight")
print("wrote", out)
