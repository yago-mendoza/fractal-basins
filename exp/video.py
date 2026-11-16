"""The social video, second version: one start, then its neighbour, then every start.

Everything drawn is measured on one GPU in one run (exp/v1_medium_traces.py, v1_pick.py and the
zoom levels of e2_zoom.py on that map):
  1. the medium Sudoku
  2. the model's starting state for one pixel, the 97 x 512 numbers it really starts from
  3. that state shrinks to a point on the plane
  4. the point loops: the decoded grid of every loop, until it stops changing; the point takes
     the colour of that loop count
  5. the pixel right next to it does the same, and takes ten more loops
  6. every pixel of the plane at once, loop by loop: settled pixels take their colour, the rest
     shimmer by how wrong their current grid is
  7. zooms into the speckle
  8. credits

usage: python exp/video.py out/v1_small.npz out/v1_zoom.npz social/fractal-basins.mp4 [sheet]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle, FancyBboxPatch
import imageio_ffmpeg

from common import SUDOKUS

SMALL, ZOOMF, MP4 = sys.argv[1:4]
FPS, SIZE = 30, 1080
BG = "#0d0d11"
INK, MUTED, FAINT = "#ececf2", "#9a9aa8", "#3a3a46"
RIGHT, WRONG, BOX = "#4a9bff", "#ff7a3d", "#5ee0f0"
FONT, LIGHT, BOLD = "Segoe UI", "Segoe UI Light", "Segoe UI Semibold"
VMAX = 20
MAGMA = matplotlib.colormaps["magma"]

d = np.load(SMALL)
SETTLE, WRONGN, CHANGED, SOL = d["settle"], d["wrong"].astype(float), d["changed"], d["sol"]
RES = SETTLE.shape[0]
KMAX = int(SETTLE.max())
ROW, CA, CB = (int(x) for x in d["pair"])
TRA, TRB = d["trace_a"], d["trace_b"]
SA, SB = int(SETTLE[ROW, CA]), int(SETTLE[ROW, CB])
ZA = d["z_a"].astype(float)
Q = SUDOKUS["medium"]
G = np.linspace(-1, 1, RES)
H = G[1] - G[0]
PA, PB = (G[CA], G[ROW]), (G[CB], G[ROW])
MID = ((PA[0] + PB[0]) / 2, G[ROW])
z = np.load(ZOOMF)
LEVELS = [(SETTLE, np.array([0.0, 0.0, 2.0]))] + [(z[f"level{k}_settle"], z[f"level{k}_box"])
                                                   for k in range(1, 5) if f"level{k}_box" in z.files]
MAXW = WRONGN.max()

fig = plt.figure(figsize=(SIZE / 100, SIZE / 100), dpi=100, facecolor=BG)


# helpers --------------------------------------------------------------------------------
def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def lerp(a, b, u):
    return a + (b - a) * u


def lerp_rect(r0, r1, u):
    return [lerp(x, y, u) for x, y in zip(r0, r1)]


def text(x, y, s, size, color=INK, alpha=1.0, font=LIGHT, ha="center"):
    if alpha > 0.01:
        fig.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, alpha=alpha, family=font)


def caption(t, lines):
    """lines: [(t_in, t_out, text)], each fading in and out over 0.4 s, at the top."""
    for t0, t1, s in lines:
        a = ease((t - t0) / 0.4) * (1 - ease((t - t1) / 0.4))
        text(.5, .935, s, 36, INK, a)


def magma(s):
    return MAGMA(np.clip(s / VMAX, 0, 1))


def sudoku(rect, shown=None, alpha=1.0, fs=14, givens_alpha=None):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 9); ax.set_ylim(9, 0); ax.axis("off")
    for i in range(81):
        r, c = divmod(i, 9)
        ga = alpha if givens_alpha is None else givens_alpha[i]
        if Q[i] != ".":
            if ga > 0.01:
                ax.add_patch(Rectangle((c, r), 1, 1, color="#20202a", lw=0, alpha=ga))
                ax.text(c + .5, r + .56, Q[i], ha="center", va="center", fontsize=fs, color=INK,
                        alpha=ga, family=BOLD)
        elif shown is not None:
            ok = shown[i] == SOL[i]
            if not ok:
                ax.add_patch(Rectangle((c, r), 1, 1, color=WRONG, alpha=.30 * alpha, lw=0))
            ax.text(c + .5, r + .56, str(shown[i]), ha="center", va="center", fontsize=fs,
                    color=RIGHT if ok else "#ffb08a", alpha=alpha, family=FONT)
    for k in range(10):
        w, col = (2.2, "#62626f") if k % 3 == 0 else (.7, "#2a2a34")
        ax.plot([k, k], [0, 9], color=col, lw=w, alpha=alpha, solid_capstyle="butt")
        ax.plot([0, 9], [k, k], color=col, lw=w, alpha=alpha, solid_capstyle="butt")
    return ax


def plane_axes(rect, view, alpha=1.0):
    """view: (cx, cy, width) in plane coordinates."""
    ax = fig.add_axes(rect)
    ax.set_facecolor("#111118")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(FAINT); s.set_alpha(alpha)
    cx, cy, w = view
    ax.set_xlim(cx - w / 2, cx + w / 2); ax.set_ylim(cy - w / 2, cy + w / 2)
    return ax


def pixel_grid(ax, view, alpha):
    """Thin lines between pixels once they are large enough to see."""
    cx, cy, w = view
    if w > 40 * H or alpha <= 0:
        return
    a = alpha * (1 - ease((w - 20 * H) / (20 * H)))
    edges = np.concatenate([G - H / 2, [G[-1] + H / 2]])
    vis = edges[(edges > cx - w) & (edges < cx + w)]
    for e in vis:
        ax.axvline(e, color="#22222c", lw=.8, alpha=a)
    vis = edges[(edges > cy - w) & (edges < cy + w)]
    for e in vis:
        ax.axhline(e, color="#22222c", lw=.8, alpha=a)


def cell(ax, p, color, alpha=1.0, ring=None, ring_alpha=0.0):
    x, y = p
    if alpha > 0:
        ax.add_patch(Rectangle((x - H / 2, y - H / 2), H, H, color=color, lw=0, alpha=alpha))
    if ring_alpha > 0:
        ax.add_patch(Rectangle((x - H / 2, y - H / 2), H, H, fill=False, ec=ring, lw=2.5,
                               alpha=ring_alpha))


def colorbar(rect, alpha=1.0, marks=()):
    if alpha <= 0.01:
        return
    ax = fig.add_axes(rect)
    ax.imshow(np.linspace(0, 1, 256)[None], cmap="magma", aspect="auto", extent=(0, VMAX, 0, 1),
              alpha=alpha)
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.patch.set_alpha(0)
    ax.tick_params(colors=MUTED, labelsize=17, length=0, pad=6)
    for lab in ax.get_xticklabels():
        lab.set_alpha(alpha); lab.set_family(FONT)
    for s in ax.spines.values():
        s.set_visible(False)
    for v, col in marks:
        ax.plot([v, v], [-0.35, 1.35], color=col, lw=3, alpha=alpha, clip_on=False)
    fig.text(rect[0], rect[1] + rect[3] + .018, "loops until the answer stops changing", fontsize=19,
             color=MUTED, alpha=alpha, family=FONT)


def state_image(rect, alpha=1.0):
    ax = fig.add_axes(rect)
    lim = np.percentile(np.abs(ZA), 99)
    ax.imshow(ZA, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest",
              alpha=alpha)
    ax.axis("off")
    return ax


def map_rgb(k):
    """The whole plane at loop k: settled pixels in their colour, the rest by how wrong."""
    s = SETTLE.ravel()
    rgb = magma(s)[:, :3]
    w = WRONGN[:, min(k, WRONGN.shape[1] - 1)] / MAXW
    ch = CHANGED[:, min(k, CHANGED.shape[1] - 1)]
    grey = (0.09 + 0.33 * w + 0.10 * ch)[:, None] * np.array([0.92, 0.95, 1.1])[None]
    out = np.where((s <= k)[:, None], rgb, grey)
    return out.reshape(RES, RES, 3)


# layout ---------------------------------------------------------------------------------
P_SMALL = [0.03, 0.17, 0.56, 0.56]        # plane, left, while the two starts run
P_FULL = [0.12, 0.13, 0.76, 0.76]         # plane, full
S_TOP = [0.625, 0.475, 0.355, 0.355]         # sudoku of the first start
S_BOT = [0.625, 0.065, 0.355, 0.355]        # sudoku of the second start
S_CENTER = [0.25, 0.2, 0.5, 0.5]
STRIP = [0.03, 0.40, 0.56, 0.11]
CBAR_SMALL = [0.03, 0.075, 0.56, 0.024]
CBAR_FULL = [0.12, 0.05, 0.76, 0.024]
FULL_VIEW = (0.0, 0.0, 2 + H)
NEAR_VIEW = (MID[0], MID[1], 9 * H)


def fig_xy(rect, view, p):
    cx, cy, w = view
    return (rect[0] + (p[0] - (cx - w / 2)) / w * rect[2],
            rect[1] + (p[1] - (cy - w / 2)) / w * rect[3])


# scenes ---------------------------------------------------------------------------------
def s_puzzle(t):
    ga = [ease((t - 0.3 - 0.018 * i) / 0.35) for i in range(81)]
    sudoku(S_CENTER, alpha=ease(t / 0.5), fs=22, givens_alpha=ga)
    caption(t, [(0.6, 99, "A model that reasons in loops gets a Sudoku.")])


def s_state(t):
    u = ease((t - 0.2) / 1.0)
    rect = lerp_rect(S_CENTER, S_TOP, u)
    sudoku(rect, fs=lerp(22, 14, u))
    a = ease((t - 1.1) / 0.6)
    if a > 0:
        ax = state_image(STRIP, a)
        wipe = ease((t - 1.1) / 1.0)
        ax.add_patch(Rectangle((wipe * 512 - .5, -.5), 512, 97, color=BG, lw=0))
        text(.31, .37, "97 x 512 numbers", 22, MUTED, a, FONT)
    caption(t, [(0.0, 99, "It starts from a state: 49,664 numbers.")])


T_STATE_TO_POINT = 3.2


def s_point(t):
    sudoku(S_TOP)
    pa = ease(t / 0.8)
    ax = plane_axes(P_SMALL, FULL_VIEW, pa)
    ax.axhline(0, color="#1d1d26", lw=1); ax.axvline(0, color="#1d1d26", lw=1)
    u = ease((t - 0.8) / 1.4)
    px, py = fig_xy(P_SMALL, FULL_VIEW, PA)
    rect = [lerp(STRIP[0], px, u), lerp(STRIP[1], py, u), lerp(STRIP[2], 0.004, u),
            lerp(STRIP[3], 0.004, u)]
    if u < 0.98:
        state_image(rect, 1 - ease((u - 0.6) / 0.4))
    g = ease((t - 1.8) / 0.5)
    for s_, a_ in ((900, .06), (380, .12), (120, .35), (30, 1)):
        ax.scatter(*PA, s=s_ * g, color="#ffffff", alpha=a_ * g, lw=0)
    text(.31, .11, "every point of this square is a possible start", 21, MUTED,
         ease((t - 2.2) / 0.5), FONT)
    caption(t, [(0.0, 99, "Think of that start as one point on a plane.")])


STEP_A, STEP_B = 0.6, 0.42


def run_start(t, which):
    """One start loops; the camera is near the pair."""
    first = which == "a"
    view = NEAR_VIEW
    zin = 1.0 if not first else ease(t / 1.4)
    if first:
        w = FULL_VIEW[2] * (NEAR_VIEW[2] / FULL_VIEW[2]) ** zin
        view = (lerp(0, MID[0], ease(t / 0.9)), lerp(0, MID[1], ease(t / 0.9)), w)
    ax = plane_axes(P_SMALL, view)
    pixel_grid(ax, view, 1.0)
    t0 = 1.5 if first else 0.6
    step = STEP_A if first else STEP_B
    tr, s, p = (TRA, SA, PA) if first else (TRB, SB, PB)
    k = int(max(t - t0, 0) / step) if t >= t0 else -1
    k = min(k, s)
    pulse = 0.5 + 0.5 * np.cos(2 * np.pi * max(t - t0, 0) / step)
    # the other start, already done (b) or not yet run (a)
    if not first:
        cell(ax, PA, magma(SA), 1.0)
        sudoku(S_TOP, TRA[SA], alpha=1.0)
        text(S_TOP[0] + S_TOP[2] / 2, S_TOP[1] - .025, f"settled at loop {SA}", 22, RIGHT, 1, BOLD)
    else:
        sudoku(S_TOP, TRA[k] if k >= 0 else None)
    done = k >= s
    if done:
        cell(ax, p, magma(s), ease((t - t0 - s * step) / 0.4))
    cell(ax, p, None, 0, ring="#ffffff", ring_alpha=(0.35 + 0.65 * pulse) if not done else 0.5)
    if not first:
        sudoku(S_BOT, tr[k] if k >= 0 else None, alpha=ease(t / 0.5))
    slot = S_TOP if first else S_BOT
    if done:
        text(slot[0] + slot[2] / 2, slot[1] - .025, f"settled at loop {s}", 22, RIGHT,
             ease((t - t0 - s * step) / 0.4), BOLD)
    elif k >= 0:
        n = int((tr[k] != SOL).sum())
        text(slot[0] + slot[2] / 2, slot[1] - .025, f"{n} cells wrong", 22, "#ffb08a", 1, FONT)
    if k >= 0:
        text(P_SMALL[0] + .02, P_SMALL[1] + P_SMALL[3] - .035, f"loop {k}", 34, INK, 1, BOLD, "left")
    marks = [(SA, "#ffffff")] if (not first or done) else []
    if not first and done:
        marks.append((SB, "#ffffff"))
    colorbar(CBAR_SMALL, ease((t - t0 - s * step) / 0.5) if first else 1.0, marks)
    return k, done


def s_run_a(t):
    k, done = run_start(t, "a")
    caption(t, [(0.0, 1.5 + (SA + 1) * STEP_A - 0.2, "Each loop, it rewrites its guess."),
                (1.5 + (SA + 1) * STEP_A, 99, "Colour the point by the loops it took.")])


def s_run_b(t):
    k, done = run_start(t, "b")
    caption(t, [(0.0, 0.6 + (SB + 1) * STEP_B, "Now the point right next to it."),
                (0.6 + (SB + 1) * STEP_B + 0.4, 99, "One pixel apart.")])


STEP_ALL = 0.42


def s_all(t):
    u = ease(t / 2.0)
    rect = lerp_rect(P_SMALL, P_FULL, u)
    w = NEAR_VIEW[2] * (FULL_VIEW[2] / NEAR_VIEW[2]) ** u
    view = (lerp(MID[0], 0, u), lerp(MID[1], 0, u), w)
    ax = plane_axes(rect, view)
    t0 = 2.3
    kf = max(t - t0, 0) / STEP_ALL
    k = int(kf)
    frac = ease((kf - k) / 0.4)
    img = map_rgb(k) * (1 - frac) + map_rgb(k + 1) * frac if t >= t0 else None
    if img is None:
        img = np.zeros((RES, RES, 3)) + 0.06
        img = img.reshape(-1, 3)
        img[ROW * RES + CA] = magma(SA)[:3]
        img[ROW * RES + CB] = magma(SB)[:3]
        img = img.reshape(RES, RES, 3)
    ax.imshow(img, origin="lower", extent=(-1 - H / 2, 1 + H / 2, -1 - H / 2, 1 + H / 2),
              interpolation="nearest")
    pixel_grid(ax, view, 1 - u)
    fade = 1 - ease(t / 0.8)
    if fade > 0:
        sudoku(S_TOP, TRA[SA], alpha=fade)
        sudoku(S_BOT, TRB[SB], alpha=fade)
    if t >= t0:
        text(rect[0] + .02, rect[1] + rect[3] - .035, f"loop {min(k, KMAX)}", 34, INK, 1, BOLD, "left")
    colorbar(lerp_rect(CBAR_SMALL, CBAR_FULL, u), 1.0)
    caption(t, [(0.2, 99, "Now every point of the plane, all at once.")])


def s_zoom(t):
    ax = plane_axes(P_FULL, FULL_VIEW)
    zt, hold = 2.5, 0.5
    n = len(LEVELS) - 1
    k = min(int(t / (zt + hold)), n - 1)
    u = ease((t - k * (zt + hold)) / zt)
    (f0, b0), (f1, b1) = LEVELS[k], LEVELS[k + 1]
    w = b0[2] * (b1[2] / b0[2]) ** u
    fr = (b0[2] - w) / (b0[2] - b1[2])
    cx, cy = lerp(b0[0], b1[0], fr), lerp(b0[1], b1[1], fr)
    for f, b, a in [(f0, b0, 1.0), (f1, b1, ease((u - .4) / .5))]:
        if a > 0:
            r = f.shape[0]
            hh = b[2] / (r - 1)
            ext = (b[0] - b[2] / 2 - hh / 2, b[0] + b[2] / 2 + hh / 2,
                   b[1] - b[2] / 2 - hh / 2, b[1] + b[2] / 2 + hh / 2)
            ax.imshow(f, origin="lower", cmap="magma", vmin=0, vmax=VMAX, extent=ext,
                      interpolation="nearest", alpha=a)
    ba = (1 - ease((u - .6) / .3)) * ease((t - k * (zt + hold) + 0.4) / 0.4)
    if ba > 0:
        ax.add_patch(Rectangle((b1[0] - b1[2] / 2, b1[1] - b1[2] / 2), b1[2], b1[2], fill=False,
                               ec=BOX, lw=2.5, alpha=ba))
    ax.set_xlim(cx - w / 2, cx + w / 2); ax.set_ylim(cy - w / 2, cy + w / 2)
    zoom = 2.0 / w
    text(P_FULL[0] + P_FULL[2] - .02, P_FULL[1] + P_FULL[3] - .035, f"{zoom:.0f}x", 34, INK,
         ease(t / 0.4) if zoom > 1.05 else 0, BOLD, "right")
    colorbar(CBAR_FULL, 1.0)
    end = n * (zt + hold)
    caption(t, [(0.0, end - 1.2, "Zoom in on the speckle."),
                (end - 0.8, 99, "Still speckle at 256x.")])


def s_end(t):
    a = ease(t / 0.7)
    text(.5, .60, "Fractal basins", 54, INK, a, BOLD)
    text(.5, .52, "in a model that reasons in loops", 30, MUTED, a)
    b = ease((t - 1.0) / 0.7)
    text(.5, .38, "reproduced from Lai, Bao, Quinn & Gilpin (2026)", 24, MUTED, b, FONT)
    text(.5, .335, "on their released models, plus my own experiments", 24, MUTED, b, FONT)
    text(.5, .22, "infraphysics.net", 32, RIGHT, b, BOLD)


SCENES = [
    (3.6, s_puzzle),
    (3.4, s_state),
    (T_STATE_TO_POINT, s_point),
    (1.5 + (SA + 1) * STEP_A + 2.2, s_run_a),
    (0.6 + (SB + 1) * STEP_B + 2.8, s_run_b),
    (2.3 + (KMAX + 1) * STEP_ALL + 1.8, s_all),
    ((len(LEVELS) - 1) * 3.0 + 1.6, s_zoom),
    (4.5, s_end),
]


def frames():
    for dur, draw in SCENES:
        for i in range(int(round(dur * FPS))):
            fig.clf()
            draw(i / FPS)
            fig.canvas.draw()
            yield np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


if __name__ == "__main__":
    only = sys.argv[4] if len(sys.argv) > 4 else None   # "sheet": a contact sheet instead
    if only == "sheet":
        from PIL import Image
        stamps, total = [], 0
        for dur, draw in SCENES:
            for f in (0.15, 0.5, 0.92):
                fig.clf(); draw(dur * f); fig.canvas.draw()
                stamps.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
                              .resize((360, 360)))
        sheet = Image.new("RGB", (360 * 6, 360 * ((len(stamps) + 5) // 6)), BG)
        for i, im in enumerate(stamps):
            sheet.paste(im, (360 * (i % 6), 360 * (i // 6)))
        sheet.save(MP4)
        print("wrote", MP4)
        sys.exit()
    w = imageio_ffmpeg.write_frames(MP4, (SIZE, SIZE), fps=FPS, codec="libx264",
                                    pix_fmt_out="yuv420p", quality=9, macro_block_size=8)
    w.send(None)
    n = 0
    for fr in frames():
        w.send(fr); n += 1
    w.close()
    print(f"wrote {MP4}: {n} frames, {n / FPS:.1f} s")
