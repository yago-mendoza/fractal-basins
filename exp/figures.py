"""Article figures (InfraPhysics Bits2Bricks 8316490), drawn from the arrays in out/.

Dark surface, magma for settle time, no titles inside the image (the captions carry the
numbers), a short label under each panel. Writes PNG masters into the site's media folder.

usage: python exp/figures.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

OUT = Path(sys.argv[1])
BG, INK, MUTED, BOX = "#101013", "#d8d8e0", "#8a8a98", "#5ec8e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def panel(ax, field, label=None, norm=None, vmax=None, extent=None):
    im = ax.imshow(field, origin="lower", cmap="magma", interpolation="nearest",
                   norm=norm, vmin=None if norm else 0, vmax=None if norm else vmax, extent=extent)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#2a2a33")
    if label:
        ax.set_xlabel(label, color=MUTED, fontsize=11, labelpad=6)
    return im


def bar(fig, im, axs, label="loops to settle"):
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.015)
    cb.outline.set_visible(False)
    cb.set_label(label, color=MUTED)
    cb.ax.tick_params(colors=MUTED)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=130, facecolor=BG, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def single(field, name, vmax):
    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor=BG)
    im = panel(ax, field, vmax=vmax)
    bar(fig, im, ax)
    save(fig, name)


e1m = np.load("out/e1_medium_200.npz")
e1h = np.load("out/e1_hard_128.npz")
single(e1m["medium_settle"], "medium-sudoku-basin", 20)
single(e1h["hard_settle"], "hard-sudoku-basin", 48)


def zoom(path, name, labels, vmax):
    d = np.load(path)
    n = 1 + max(int(k[5:].split("_")[0]) for k in d.files if k.startswith("level"))
    fig, axs = plt.subplots(1, n, figsize=(4.0 * n, 4.6), facecolor=BG)
    for lv, ax in enumerate(axs):
        cx, cy, w = d[f"level{lv}_box"]
        ext = (cx - w / 2, cx + w / 2, cy - w / 2, cy + w / 2)
        im = panel(ax, d[f"level{lv}_settle"], labels[lv], vmax=vmax, extent=ext)
        if lv + 1 < n:
            nx, ny, nw = d[f"level{lv + 1}_box"]
            ax.add_patch(Rectangle((nx - nw / 2, ny - nw / 2), nw, nw, fill=False, ec=BOX, lw=1.6))
    bar(fig, im, list(axs))
    save(fig, name)


zoom("out/e2_medium_zoom.npz", "medium-zoom", ["whole slice", "4x", "16x", "64x"], 23)
zoom("out/e2_hard_zoom.npz", "hard-zoom", ["whole slice", "10x", "100x"], 48)


def linsys(path, name, label_of):
    d = np.load(path)
    tags = [k[:-7] for k in d.files if k.endswith("_settle")]
    vmax = max(d[t + "_settle"].max() for t in tags)
    fig, axs = plt.subplots(1, len(tags), figsize=(2.9 * len(tags), 3.6), facecolor=BG, squeeze=False)
    for ax, t in zip(axs[0], tags):
        ok = d[t + "_metrics"][0]
        im = panel(ax, np.clip(d[t + "_settle"], 1, None), f"{label_of(t)}\n{'right' if ok > 0.5 else 'wrong'}",
                   norm=LogNorm(1, vmax))
    bar(fig, im, list(axs[0]), "loops to settle (log)")
    save(fig, name)


lab = lambda t: "released" if t == "released" else f"step {int(t[4:]) // 1000}k"
linsys("out/e3_linsys.npz", "linsys-training", lab)
linsys("out/e3_linsys_easy.npz", "linsys-easy-instance", lambda t: "released" if t == "released" else "ours, step 60k")
