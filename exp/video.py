"""A short square video for social posts (LinkedIn, X), drawn from the arrays in out/.

  1. title
  2. the same hard Sudoku from two random starts, decoded loop by loop (E6 traces)
  3. the medium Sudoku's settle-time map, then a continuous zoom to 64x (E2 levels)
  4. credits

Needs out/thinking_loops_pick.npz (from sudoku_figures.py) and the held-out puzzle text, so it
takes the same CSV.

usage: python exp/video.py TEST_CSV OUT.mp4 [OUT.gif]
"""
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio_ffmpeg

CSV, MP4 = sys.argv[1], sys.argv[2]
GIF = sys.argv[3] if len(sys.argv) > 3 else None
FPS, SIZE = 30, 1080
BG, INK, MUTED, LINE, THIN = "#101013", "#e4e4ec", "#8a8a98", "#6a6a78", "#2c2c36"
RIGHT, WRONG, BOX = "#3987e5", "#d95926", "#5ec8e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})

# data -------------------------------------------------------------------------------------
pick = np.load("out/thinking_loops_pick.npz")
P, SA, SB = int(pick["puzzle"]), int(pick["fast"]), int(pick["slow"])
d6 = np.load("out/e6_starts.npz")
rows = [r for r in csv.DictReader(open(CSV)) if int(r["rating"]) >= 100]
rng = np.random.default_rng(0)
test = [rows[i] for i in rng.choice(len(rows), 4 + d6["traces"].shape[0], replace=False)][4:]
Q, SOL = test[P]["question"], d6["sols"][P]
TA, TB = d6["traces"][P, SA], d6["traces"][P, SB]
assert np.array_equal(np.array([int(c) for c in test[P]["answer"]]), SOL)


def solved_at(tr):
    diff = np.nonzero((tr != tr[-1]).any(-1))[0]
    return (0 if len(diff) == 0 else int(diff[-1]) + 1) + 1        # 1-based loop


LA, LB = solved_at(TA), solved_at(TB)
z = np.load("out/e2_medium_zoom.npz")
LEVELS = [(z[f"level{k}_settle"], z[f"level{k}_box"]) for k in range(4)]
VMAX = 23

fig = plt.figure(figsize=(SIZE / 100, SIZE / 100), dpi=100, facecolor=BG)


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def text(x, y, s, size, color=INK, alpha=1.0, weight="normal"):
    fig.text(x, y, s, ha="center", va="center", fontsize=size, color=color, alpha=alpha,
             fontweight=weight)


def sudoku(rect, shown, alpha=1.0):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 9); ax.set_ylim(9, 0); ax.set_aspect("equal"); ax.axis("off")
    for i in range(81):
        r, c = divmod(i, 9)
        if Q[i] != ".":
            ax.add_patch(Rectangle((c, r), 1, 1, color="#22222b", lw=0, alpha=alpha))
            ax.text(c + .5, r + .55, Q[i], ha="center", va="center", fontsize=17, color=INK,
                    fontweight="bold", alpha=alpha)
        elif shown is not None:
            ok = shown[i] == SOL[i]
            if not ok:
                ax.add_patch(Rectangle((c, r), 1, 1, color=WRONG, alpha=.32 * alpha, lw=0))
            ax.text(c + .5, r + .55, str(shown[i]), ha="center", va="center", fontsize=17,
                    color=RIGHT if ok else "#f3a782", alpha=alpha)
    for k in range(10):
        w, col = (2.4, LINE) if k % 3 == 0 else (.8, THIN)
        ax.plot([k, k], [0, 9], color=col, lw=w, alpha=alpha, solid_capstyle="butt")
        ax.plot([0, 9], [k, k], color=col, lw=w, alpha=alpha, solid_capstyle="butt")


# scenes: each is (seconds, draw(t) with t in seconds from the scene start) ---------------
def s_title(t):
    a = ease(t / 0.8)
    text(.5, .58, "A model that thinks in loops", 40, alpha=a, weight="bold")
    text(.5, .48, "It applies one block to its own state again and again,", 22, MUTED, a)
    text(.5, .43, "and after every loop it decodes its current answer.", 22, MUTED, a)


STEP = 0.42                                    # seconds per loop in the race
RACE_LOOPS = LB + 3


def s_race(t):
    L = min(1 + int(t / STEP), RACE_LOOPS)
    text(.5, .92, "Same hard Sudoku, same model. Two random starting points.", 23)
    for x0, tr, name, done in [(.05, TA, "start A", LA), (.53, TB, "start B", LB)]:
        shown = tr[L - 1]
        sudoku([x0, .34, .42, .42], shown)
        text(x0 + .21, .80, name, 22, MUTED)
        n = int((shown != SOL).sum())
        if n:
            text(x0 + .21, .29, f"{n} cells wrong", 22, "#f3a782")
        else:
            text(x0 + .21, .29, f"solved at loop {done}", 22, RIGHT, weight="bold")
    text(.5, .17, f"loop {L}", 46, weight="bold")
    if L >= RACE_LOOPS:
        a = ease((t - RACE_LOOPS * STEP) / 0.6)
        text(.5, .08, f"B wandered for {LB - LA} more loops, then the whole grid fell into place at once.",
             19, MUTED, a)


def s_map(t):
    """3 s on the whole map, then three zooms of 4x, 2.6 s each, 0.7 s holds."""
    ax = fig.add_axes([.09, .17, .82, .66])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#2a2a33")
    intro, zt, hold = 3.0, 2.6, 0.7
    k, u = 0, 0.0
    if t > intro:
        k = min(int((t - intro) / (zt + hold)), 2)
        u = ease(((t - intro) - k * (zt + hold)) / zt)
    (f0, b0), (f1, b1) = LEVELS[k], LEVELS[k + 1]
    w = b0[2] * (b1[2] / b0[2]) ** u
    frac = (b0[2] - w) / (b0[2] - b1[2])
    cx, cy = b0[0] + (b1[0] - b0[0]) * frac, b0[1] + (b1[1] - b0[1]) * frac
    for f, b, alpha in [(f0, b0, 1.0), (f1, b1, ease((u - .45) / .5))]:
        if alpha > 0:
            ext = (b[0] - b[2] / 2, b[0] + b[2] / 2, b[1] - b[2] / 2, b[1] + b[2] / 2)
            ax.imshow(f, origin="lower", cmap="magma", vmin=0, vmax=VMAX, extent=ext,
                      interpolation="nearest", alpha=alpha)
    if t > intro - 0.8 and u < .98:
        a = ease((t - (intro - 0.8)) / 0.6) * (1 - ease((u - .7) / .25))
        ax.add_patch(Rectangle((b1[0] - b1[2] / 2, b1[1] - b1[2] / 2), b1[2], b1[2], fill=False,
                               ec=BOX, lw=2.5, alpha=a))
    ax.set_xlim(cx - w / 2, cx + w / 2); ax.set_ylim(cy - w / 2, cy + w / 2)
    zoom = LEVELS[0][1][2] / w
    a = ease(t / 0.6)
    text(.5, .92, "Now do it from 40,000 starting points on a plane,", 23, alpha=a)
    text(.5, .875, "and colour each one by how many loops it needs.", 23, alpha=a)
    text(.5, .11, "medium Sudoku   dark: settles fast   bright: thinks long", 18, MUTED, a)
    text(.5, .05, f"zoom {zoom:.0f}x" if zoom > 1.05 else "the whole slice", 26,
         weight="bold", alpha=a)


def s_tail(t):
    a = ease(t / 0.6)
    text(.5, .66, "The speckle survives a 64x zoom.", 30, alpha=a, weight="bold")
    text(.5, .58, "Two starts a hair apart can need very different thinking.", 22, MUTED, a)
    b = ease((t - 1.2) / 0.6)
    text(.5, .42, "Reproduction of Lai, Bao, Quinn & Gilpin (2026)", 20, MUTED, b)
    text(.5, .375, "on their released models, plus my own experiments.", 20, MUTED, b)
    text(.5, .25, "infraphysics.net", 30, RIGHT, b, weight="bold")


SCENES = [(3.0, s_title), (RACE_LOOPS * STEP + 3.2, s_race), (3.0 + 3 * 3.3 + 0.8, s_map),
          (4.5, s_tail)]


def frames(scenes, fps):
    for dur, draw in scenes:
        n = int(round(dur * fps))
        for i in range(n):
            fig.clf()
            draw(i / fps)
            fig.canvas.draw()
            yield np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


w = imageio_ffmpeg.write_frames(MP4, (SIZE, SIZE), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                quality=8, macro_block_size=8)
w.send(None)
count = 0
for fr in frames(SCENES, FPS):
    w.send(fr)
    count += 1
w.close()
print(f"wrote {MP4}: {count} frames, {count / FPS:.1f} s")

if GIF:   # the race alone, small, for the README
    from PIL import Image
    imgs = [Image.fromarray(fr).resize((540, 540), Image.LANCZOS)
            for fr in frames([(RACE_LOOPS * STEP + 3.2, s_race)], 10)]
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in imgs]
    pal[0].save(GIF, save_all=True, append_images=pal[1:], duration=100, loop=0, optimize=True)
    print("wrote", GIF)
