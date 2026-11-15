"""Quick look at an E1 npz: settle-time field per puzzle, plus solved mask where it varies."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(".npz", ".png")
names = [k[:-7] for k in d.files if k.endswith("_settle")]
fig, axs = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.4), squeeze=False)
for ax, n in zip(axs[0], names):
    s = d[f"{n}_settle"]
    im = ax.imshow(s, origin="lower", cmap="magma", interpolation="nearest")
    sb, sbb, a = d[f"{n}_metrics"]
    ax.set_title(f"{n}: solved {d[n + '_solved'].mean():.0%}\nSb={sb:.2f}  alpha={a:.2f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, label="loops to settle")
fig.tight_layout()
fig.savefig(out, dpi=110)
print("wrote", out)
