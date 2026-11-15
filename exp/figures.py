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
ALL = len(sys.argv) <= 2
if ALL:
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


if ALL:
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
if ALL:
    linsys("out/e3_linsys.npz", "linsys-training", lab)
    linsys("out/e3_linsys_easy.npz", "linsys-easy-instance", lambda t: "released" if t == "released" else "ours, step 60k")


def saddles(path, name):
    """E4: wrong cells and latent step size per loop, fast vs slow starts (median, IQR, a few traces)."""
    d = np.load(path)
    col = {"fast": "#3987e5", "slow": "#d95926"}
    lab = {"fast": "fast starts (smooth region)", "slow": "slow starts (speckle)"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=BG)
    for ax, key, ylab, logy in [(axs[0], "wrong", "cells different from the solution", False),
                                (axs[1], "step", "how far the latent state moves", True)]:
        ax.set_facecolor(BG)
        for g in ("fast", "slow"):
            y = d[f"{g}_{key}"].astype(float)
            x = np.arange(1, y.shape[1] + 1)
            for tr in y[:6]:
                ax.plot(x, tr, color=col[g], lw=0.8, alpha=0.25)
            q1, med, q3 = np.percentile(y, [25, 50, 75], axis=0)
            ax.fill_between(x, q1, q3, color=col[g], alpha=0.18, lw=0)
            ax.plot(x, med, color=col[g], lw=2, label=lab[g])
        ax.set_xlabel("loop"); ax.set_ylabel(ylab)
        if logy:
            ax.set_yscale("log")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#3a3a44")
        ax.grid(axis="y", color="#26262e", lw=0.6)
    leg = axs[0].legend(frameon=False, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, name)


if len(sys.argv) > 2 and sys.argv[2] == "e4":
    saddles("out/e4_saddles.npz", "slow-starts")


def policies(path, name, k=3):
    """E6: share of held-out hard Sudokus answered right against the loop budget, per policy."""
    import json
    res = json.load(open(path))
    budgets = sorted({int(key.split("_B")[1]) for key in res if key.startswith(f"k{k}_B")})
    series = [("wait", "one random start, wait", "#8a8a98"),
              ("restart20", "restart if not settled by loop 20", "#3987e5"),
              ("evolved+restart30", "evolved start, then restarts", "#d95926")]
    fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG)
    ax.set_facecolor(BG)
    for key, label, c in series:
        bs = [b for b in budgets if key != "wait" or b <= 64]   # one start's trace ends at loop 64
        y = [100 * res[f"k{k}_B{b}"][key][0] for b in bs]
        ax.plot(bs, y, color=c, lw=2, marker="o", ms=6, label=label)
        ax.annotate(f"{y[-1]:.1f}%", (bs[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    color=INK, va="center", fontsize=10)
    ax.set_ylim(90, 100.5)
    ax.set_xlabel("loop budget per puzzle"); ax.set_ylabel("puzzles answered right (%)")
    ax.set_xticks(budgets)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#3a3a44")
    ax.grid(axis="y", color="#26262e", lw=0.6)
    leg = ax.legend(frameon=False, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, name)


if len(sys.argv) > 2 and sys.argv[2] == "e6":
    policies("out/e6_policies.json", "start-policies")


def difficulty(path, name):
    """E5: classical difficulty (tdoku backtracks) against how long EqR takes, one dot per puzzle."""
    d = np.load(path)
    r, st, un = d["rating"].astype(float), d["settle_mean"], d["unsettled"]
    lost = un > 0.05
    fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.scatter(r[~lost], st[~lost], s=46, color="#3987e5", edgecolor=BG, lw=1.5,
               label="settles from (almost) every start")
    ax.scatter(r[lost], st[lost], s=46, color="#d95926", edgecolor=BG, lw=1.5,
               label="more than 5% of starts never settle")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 3, 10, 30, 100, 300]); ax.set_xticklabels(["0", "1", "3", "10", "30", "100", "300"])
    ax.set_xlabel("backtracks the tdoku solver needs (Sudoku-Extreme rating)")
    ax.set_ylabel("mean loops for EqR to settle")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#3a3a44")
    ax.grid(axis="y", color="#26262e", lw=0.6)
    leg = ax.legend(frameon=False, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, name)


if len(sys.argv) > 2 and sys.argv[2] == "e5":
    difficulty("out/e5_difficulty.npz", "classical-difficulty")
