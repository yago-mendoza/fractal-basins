"""Two figures that show the Sudokus themselves, before any map.

  three-sudokus   the three puzzles of E1 as grids (givens only)
  thinking-loops  one held-out hard puzzle from E6, the grid EqR decodes after each loop,
                  for a start that settles at once and one that wanders first

The E6 traces keep the decoded grids but not the puzzles, so the held-out puzzles are picked
again from the Sudoku-Extreme test split exactly as e6_starts.py picks them, and checked
against the stored solutions.

usage: python exp/sudoku_figures.py TEST_CSV OUT_DIR
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from common import SUDOKUS  # noqa: E402  (torch is imported there too; only the strings are used)

CSV, OUT = sys.argv[1], Path(sys.argv[2])
BG, INK, MUTED, LINE, THIN = "#101013", "#d8d8e0", "#8a8a98", "#6a6a78", "#2c2c36"
RIGHT, WRONG = "#3987e5", "#d95926"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})


def grid(ax, given, shown=None, sol=None, fs=9):
    """Draw a 9x9 grid. given: 81 chars ('.' empty). shown: 81 digits decoded by the model."""
    ax.set_xlim(0, 9); ax.set_ylim(9, 0); ax.set_aspect("equal"); ax.axis("off")
    for i in range(81):
        r, c = divmod(i, 9)
        if given[i] != ".":
            ax.add_patch(Rectangle((c, r), 1, 1, color="#1f1f27", lw=0))
            ax.text(c + .5, r + .54, given[i], ha="center", va="center", fontsize=fs, color=INK,
                    fontweight="bold")
        elif shown is not None:
            ok = shown[i] == sol[i]
            if not ok:
                ax.add_patch(Rectangle((c, r), 1, 1, color=WRONG, alpha=.28, lw=0))
            ax.text(c + .5, r + .54, str(shown[i]), ha="center", va="center", fontsize=fs,
                    color=RIGHT if ok else "#f0a07a")
    for k in range(10):
        w, col = (1.4, LINE) if k % 3 == 0 else (.5, THIN)
        ax.plot([k, k], [0, 9], color=col, lw=w, solid_capstyle="butt")
        ax.plot([0, 9], [k, k], color=col, lw=w, solid_capstyle="butt")


def save(fig, name, dpi):
    fig.savefig(OUT / f"{name}.png", dpi=dpi, facecolor=BG, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# 1. the three puzzles
fig, axs = plt.subplots(1, 3, figsize=(11, 4.1), facecolor=BG)
for ax, key in zip(axs, ("easy", "medium", "hard")):
    q = SUDOKUS[key]
    grid(ax, q, fs=12)
    ax.set_title(f"{key}, {sum(ch != '.' for ch in q)} givens", color=MUTED, fontsize=12, pad=8)
save(fig, "three-sudokus", 160)

# 2. the held-out puzzles of E6, picked as e6_starts.py picks them
rows = [r for r in csv.DictReader(open(CSV)) if int(r["rating"]) >= 100]
rng = np.random.default_rng(0)
d = np.load("out/e6_starts.npz")
P = d["traces"].shape[0]
pick = [rows[i] for i in rng.choice(len(rows), 4 + P, replace=False)]
test = pick[4:]
assert [int(r["rating"]) for r in test] == d["ratings"].tolist(), "ratings differ from E6"
for r, s in zip(test, d["sols"]):
    assert np.array_equal(np.array([int(c) for c in r["answer"]]), s), "solution differs from E6"
print("held-out puzzles reconstructed and checked against E6")

TR, SOL = d["traces"], d["sols"]
K = TR.shape[1] - 2


def last_change(tr):
    diff = np.nonzero((tr != tr[-1]).any(-1))[0]
    return 0 if len(diff) == 0 else int(diff[-1]) + 1


best = None
for p in range(P):
    st = np.array([last_change(TR[p, s]) for s in range(K)])
    ok = (TR[p, :K, -1] == SOL[p]).all(-1)
    fast = [s for s in range(K) if ok[s] and st[s] <= 3]
    slow = [s for s in range(K) if ok[s] and 14 <= st[s] <= 20]
    if not fast or not slow:
        continue
    s = max(slow, key=lambda s: (TR[p, s, st[s] - 3] != SOL[p]).sum())   # still far off just before
    wrong_before = int((TR[p, s, st[s] - 3] != SOL[p]).sum())
    if best is None or wrong_before > best[3]:
        best = (p, fast[0], s, wrong_before, int(st[fast[0]]), int(st[s]))
p, sf, ss, _, tf, ts = best
print(f"puzzle {p} rating {d['ratings'][p]}: fast start {sf} settles at {tf}, slow start {ss} at {ts}")

q, sol = test[p]["question"], SOL[p]
# last_change is the 0-based index of the last grid that differs, so loop last_change + 1 is the
# first right one: show the first loops, the fast start landing, the wait, and the capture
loops = sorted({1, 2, 3, tf + 1, 8, 12, ts, ts + 1})
fig, axs = plt.subplots(2, len(loops), figsize=(1.75 * len(loops), 4.3), facecolor=BG)
for row, (s, label) in enumerate([(sf, "fast start"), (ss, "slow start")]):
    for ax, L in zip(axs[row], loops):
        shown = TR[p, s, L - 1]
        grid(ax, q, shown, sol, fs=5.2)
        n = int((shown != sol).sum())
        ax.text(4.5, 9.75, f"loop {L}" + ("" if n == 0 else f", {n} wrong"),
                ha="center", va="top", fontsize=8, color=MUTED if n else RIGHT)
    axs[row][0].text(-0.6, 4.5, label, rotation=90, ha="right", va="center", fontsize=10, color=MUTED)
fig.subplots_adjust(wspace=.08, hspace=.32)
save(fig, "thinking-loops", 200)
np.savez("out/thinking_loops_pick.npz", puzzle=p, fast=sf, slow=ss, loops=loops)
