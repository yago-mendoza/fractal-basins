"""Hero strip for the README and the article: the three E1 Sudoku fields side by side,
same colour scale per panel, no axes. Data from out/ (E1 runs)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/hero.png"
# The easy run (40x40, see README) settled at loop 0 at every point; its npz stayed on the pod.
easy = np.zeros((40, 40), dtype=int)
medium = np.load("out/e1_medium_200.npz")["medium_settle"]
hard = np.load("out/e1_hard_128.npz")["hard_settle"]
bg = "#0b0b10"
fig, axs = plt.subplots(1, 3, figsize=(12, 4.3), facecolor=bg)
for ax, field, label in zip(axs, (easy, medium, hard),
                            ("easy Sudoku: settled at once", "medium Sudoku: folds", "hard Sudoku: noise")):
    vmax = max(int(field.max()), 1)
    ax.imshow(field, origin="lower", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(label, color="#d8d8e0", fontsize=12, pad=8, family="monospace")
fig.text(0.5, 0.03, "loops until the answer stops changing, over a 2D slice of the model's latent space"
         "  (EqR, colour scaled per panel)", ha="center", color="#8a8a98", fontsize=9, family="monospace")
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.09, wspace=0.03)
fig.savefig(out, dpi=150, facecolor=bg)
print("wrote", out)
