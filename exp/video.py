"""The social video: how to read a basin map, built up one idea at a time.

Every number, grid and map shown is measured (one run on one GPU: exp/v1_medium_traces.py,
exp/v1_pick.py, then exp/e2_zoom.py on that map). Two things are diagrams and say so on screen:
the loop schematic (its tiles are decoration) and the smooth disc used to show what an ordinary
edge does under a zoom. Colours of loop counts use magma on a square-root-like scale so that the
short counts, most of the map, are not lost in the dark.

  hook        the finished map: each pixel is the same network on the same Sudoku, another start
  puzzle      the Sudoku, and a small network trained only on Sudoku (no chatbot)
  loop        puzzle and memory go into the network, memory comes out; read out as a guess
  settle      one start, loop by loop; it wanders, then snaps into place: solved, 14 loops
  start       the starting memory; many starts as points on a map; position = two dials
  colour      colour that point by its 14 loops (time, not right or wrong)
  neighbour   one pixel over: the starts differ by under 3%, and it settles at loop 2
  all         every pixel at once, loop by loop
  read        smooth areas, tangled bands, where the two starts sit
  zoom        an ordinary edge straightens when zoomed; this one still mixes at 256x
  end         what that means, the practical upshot (restarts, measured in E6), why beyond Sudoku

usage: python exp/video.py out/v1_small.npz out/v1_zoom.npz out/v1_plane.npz OUT [sheet STEP|pages STEP|preview]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
import imageio_ffmpeg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SUDOKUS  # noqa: E402

SMALL, ZOOMF, PLANEF, OUT = sys.argv[1:5]
MODE = sys.argv[5] if len(sys.argv) > 5 else "video"
FPS, SIZE = 30, 1080

# look ----------------------------------------------------------------------------------
INK, MUTED, DIM = "#f4f5fa", "#b8bdcc", "#8a90a4"
CYAN, HOT, CARD, EDGE = "#38c6ff", "#ff6b3d", "#12141d", "#272b3a"
GIVEN = "#2c3040"
HEAD, BODY = "Segoe UI Semibold", "Segoe UI"
MAGMA = matplotlib.colormaps["magma"]
MEMCMAP = LinearSegmentedColormap.from_list("mem", ["#62a8ff", "#24406e", "#0e1016", "#6a2a70", "#f07cff"])
MEMLIM = 0.0035
GAMMA = 0.55
yy, xx = np.mgrid[0:SIZE, 0:SIZE] / SIZE
_r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.42) ** 2)
_g = np.clip(1 - _r / 0.85, 0, 1)[..., None]
BGIMG = (np.array([7, 8, 12]) + _g * np.array([16, 20, 36])).astype(np.uint8)

# data ----------------------------------------------------------------------------------
d = np.load(SMALL)
SETTLE, WRONGN, CHANGED, SOL = d["settle"], d["wrong"].astype(float), d["changed"], d["sol"]
RES = SETTLE.shape[0]
KMAX = int(SETTLE.max())
VMAX = KMAX
ROW, CA, CB = (int(x) for x in d["pair"])
T1, T2 = d["trace_b"], d["trace_a"]          # T1: the first start (slow), T2: its neighbour (fast)
S1, S2 = int(SETTLE[ROW, CB]), int(SETTLE[ROW, CA])
P1, P2 = tuple(d["pts"][1]), tuple(d["pts"][0])   # P2 is the pixel to the left of P1
ZB, ZA = d["z_b"].astype(float), d["z_a"].astype(float)
pl = np.load(PLANEF)
U, V = pl["u"].astype(float), pl["v"].astype(float)
REL = np.linalg.norm(ZA - ZB) / np.linalg.norm(ZB)
Q = SUDOKUS["medium"]
G = np.linspace(-1, 1, RES)
H = G[1] - G[0]
MID = ((P1[0] + P2[0]) / 2, P1[1])
z = np.load(ZOOMF)
LEVELS = [(SETTLE, np.array([0.0, 0.0, 2.0]))] + [(z[f"level{k}_settle"], z[f"level{k}_box"])
                                                   for k in range(1, 9) if f"level{k}_box" in z.files]
DEEPEST = 2.0 / LEVELS[-1][1][2]


def magma(s):
    return MAGMA(np.clip(np.asarray(s, float) / VMAX, 0, 1) ** GAMMA)


def boxmean(a, k):
    p = np.pad(a, k // 2, mode="edge")
    c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / k ** 2


_s = SETTLE.astype(float)
VAR = boxmean(_s ** 2, 9) - boxmean(_s, 9) ** 2
_inner = np.zeros_like(VAR, bool)
_inner[30:-30, 30:-30] = True
_calm = boxmean((VAR < 0.05).astype(float), 31)
r_, c_ = np.unravel_index(np.argmax(np.where(_inner, _calm - 0.01 * _s, -1)), VAR.shape)
SMOOTH_AT = (G[c_], G[r_])
r_, c_ = np.unravel_index(np.argmax(np.where(_inner, boxmean(VAR, 15), -1)), VAR.shape)
TANGLE_AT = (G[c_], G[r_])

# two side-by-side pixels at the deepest zoom that disagree the most, near the middle
_f, _b = LEVELS[-1]
_n = _f.shape[0]
_diff = np.abs(_f[:, 1:].astype(int) - _f[:, :-1].astype(int))
_m = np.zeros_like(_diff, bool)
_m[_n // 4:3 * _n // 4, _n // 4:3 * _n // 4] = True
_rr, _cc = np.unravel_index(np.argmax(np.where(_m, _diff, -1)), _diff.shape)
_gz = np.linspace(_b[0] - _b[2] / 2, _b[0] + _b[2] / 2, _n)
_gy = np.linspace(_b[1] - _b[2] / 2, _b[1] + _b[2] / 2, _n)
DEEP_PAIR = [((_gz[_cc], _gy[_rr]), int(_f[_rr, _cc])), ((_gz[_cc + 1], _gy[_rr]), int(_f[_rr, _cc + 1]))]
DEEP_H = _b[2] / (_n - 1)

fig = plt.figure(figsize=(SIZE / 100, SIZE / 100), dpi=100)


# helpers -------------------------------------------------------------------------------
def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def eout(u):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


def lerp(a, b, u):
    return a + (b - a) * u


def lerp_rect(r0, r1, u):
    return [lerp(x, y, u) for x, y in zip(r0, r1)]


def window(t, t0, t1=None, fin=0.35, fout=0.35):
    a = ease((t - t0) / fin)
    if t1 is not None:
        a *= 1 - ease((t - t1) / fout)
    return a


_W = {}
LABEL_BOX = dict(boxstyle="round,pad=0.25", fc="#0d0f16", ec="none", alpha=0.75)


def word_w(w, size, font):
    key = (w, size, font)
    if key not in _W:
        tx = fig.text(0, 0, w, fontsize=size, family=font)
        _W[key] = tx.get_window_extent(fig.canvas.get_renderer()).width / SIZE
        tx.remove()
    return _W[key]


def kline(s, y, size, t, t0, t1=None, font=HEAD, color=INK, x=0.5, ha="center"):
    """A line whose words rise in one after another; *word* is drawn in cyan, _word_ in orange."""
    words = s.split()
    sp = word_w("x x", size, font) - 2 * word_w("x", size, font)
    clean = [w.replace("*", "").strip("_") for w in words]
    widths = [word_w(w, size, font) for w in clean]
    total = sum(widths) + sp * (len(words) - 1)
    cx = x - total / 2 if ha == "center" else x
    for i, (w, c, wd) in enumerate(zip(words, clean, widths)):
        a = eout((t - t0 - 0.05 * i) / 0.35)
        if t1 is not None:
            a *= 1 - ease((t - t1) / 0.3)
        if a > 0.01:
            col = CYAN if w.startswith("*") else HOT if w.startswith("_") else color
            fig.text(cx, y - 0.014 * (1 - a), c, fontsize=size, family=font, color=col, alpha=a,
                     ha="left", va="center")
        cx += wd + sp


def caption(t, beats, y=0.928, size=33):
    """beats: [(t0, t1, line or (line1, line2))] at the top of the frame."""
    for t0, t1, s in beats:
        lines = (s,) if isinstance(s, str) else s
        for j, ln in enumerate(lines):
            kline(ln, y - 0.052 * j, size, t, t0 + 0.12 * j, t1)


def overlay():
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_autoscale_on(False)
    return ax


def card(ov, rect, alpha=1.0, glow=None):
    if alpha <= 0.01:
        return
    x, y, w, h = rect
    if glow:
        for k, a in ((0.02, .08), (0.011, .14)):
            ov.add_patch(FancyBboxPatch((x - 0.012 - k, y - 0.012 - k), w + 0.024 + 2 * k, h + 0.024 + 2 * k,
                                        boxstyle="round,pad=0,rounding_size=0.03", fc=glow, ec="none",
                                        alpha=a * alpha))
    ov.add_patch(FancyBboxPatch((x - 0.012, y - 0.012), w + 0.024, h + 0.024,
                                boxstyle="round,pad=0,rounding_size=0.025", fc=CARD, ec=EDGE, lw=1.2,
                                alpha=alpha))


def chip(ov, x, y, color, label, alpha=1.0, size=23, outline=False):
    if alpha <= 0.01:
        return
    if outline:
        ov.add_patch(Rectangle((x + 0.002, y - 0.012), 0.022, 0.024, fill=False, ec=color, lw=2.4, alpha=alpha))
    else:
        ov.add_patch(FancyBboxPatch((x, y - 0.013), 0.026, 0.026, boxstyle="round,pad=0,rounding_size=0.005",
                                    fc=color, ec="none", alpha=alpha))
    fig.text(x + 0.04, y, label, fontsize=size, family=BODY, color=MUTED, alpha=alpha, va="center")


def sudoku(rect, shown=None, prev=None, flash=0.0, alpha=1.0, fs=15, givens=None):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 9); ax.set_ylim(9, 0); ax.axis("off")
    for i in range(81):
        r, c = divmod(i, 9)
        ga = alpha if givens is None else givens[i] * alpha
        if Q[i] != ".":
            if ga > 0.01:
                ax.add_patch(Rectangle((c + .04, r + .04), .92, .92, color=GIVEN, lw=0, alpha=ga))
                ax.text(c + .5, r + .56, Q[i], ha="center", va="center", fontsize=fs, color=INK,
                        alpha=ga, family=HEAD)
        elif shown is not None:
            ok = shown[i] == SOL[i]
            if not ok:
                ax.add_patch(Rectangle((c + .04, r + .04), .92, .92, color=HOT, alpha=.30 * alpha, lw=0))
            ax.text(c + .5, r + .56, str(shown[i]), ha="center", va="center", fontsize=fs,
                    color=CYAN if ok else "#ffb394", alpha=alpha, family=BODY)
            if prev is not None and flash > 0 and prev[i] != shown[i]:
                ax.add_patch(Rectangle((c + .06, r + .06), .88, .88, fill=False, ec="#ffffff", lw=2.4,
                                       alpha=flash * alpha))
    for k in range(10):
        w, col = (2.0, "#5d6275") if k % 3 == 0 else (.6, "#252937")
        ax.plot([k, k], [0, 9], color=col, lw=w, alpha=alpha, solid_capstyle="round")
        ax.plot([0, 9], [k, k], color=col, lw=w, alpha=alpha, solid_capstyle="round")
    return ax


def run_view(rect, trace, s, k, kf, alpha=1.0, fs=15):
    """A sudoku at loop k (k < 0: givens only), outlining the cells that changed this loop."""
    shown = trace[min(k, len(trace) - 1)] if k >= 0 else None
    prev = trace[k - 1] if 1 <= k < len(trace) else None
    flash = (1 - eout((kf - k) / 0.8)) if 1 <= k <= s else 0
    sudoku(rect, shown, prev, flash, alpha, fs)


def legend(ov, x, y, alpha, changed=False, given=False, size=23):
    rows = [(CYAN, "right", False), (HOT, "wrong", False)]
    if given:
        rows.append((GIVEN, "given", False))
    if changed:
        rows.append(("#ffffff", "just changed", True))
    for i, (col, lab, out) in enumerate(rows):
        chip(ov, x, y - 0.05 * i, col, lab, alpha, size, outline=out)


def pills(ov, x0, y, w, trace, k, n, alpha=1.0):
    """One pill per loop: white while the guess changed at that loop."""
    if alpha <= 0.01:
        return
    gap = w / n
    for i in range(n):
        if i > k:
            fc = "#1a1d28"
        elif i == 0 or (trace[i] != trace[i - 1]).any():
            fc = "#e8eaf2"
        else:
            fc = "#2f3446"
        ov.add_patch(FancyBboxPatch((x0 + i * gap + gap * .12, y - 0.011), gap * .76, 0.022,
                                    boxstyle="round,pad=0,rounding_size=0.006", fc=fc, ec="none", alpha=alpha))
    fig.text(x0, y - 0.045, "loop 0", fontsize=20, family=BODY, color=DIM, alpha=alpha)
    fig.text(x0 + w, y - 0.045, f"loop {n - 1}", fontsize=20, family=BODY, color=DIM, alpha=alpha, ha="right")
    fig.text(x0 + w / 2, y + 0.042, "white bar: the guess changed that loop", fontsize=22, family=BODY,
             color=MUTED, alpha=alpha, ha="center")


def counter(x, y, k, alpha=1.0, ha="left", size=48):
    if alpha <= 0.01 or k < 0:
        return
    fig.text(x, y + 0.052, "LOOP", fontsize=20, family=HEAD, color=MUTED, alpha=alpha, ha=ha)
    fig.text(x, y, f"{k}", fontsize=size, family=HEAD, color=INK, alpha=alpha, ha=ha, va="center")


def plane(rect, view, alpha=1.0, ticks=False):
    ax = fig.add_axes(rect)
    ax.set_facecolor("#0f1118")
    ax.patch.set_alpha(alpha)
    if ticks:
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.tick_params(colors=MUTED, labelsize=20, length=0, pad=6)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_alpha(alpha); lab.set_family(BODY)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(EDGE); s.set_alpha(alpha); s.set_linewidth(1.2)
    cx, cy, w = view
    ax.set_xlim(cx - w / 2, cx + w / 2); ax.set_ylim(cy - w / 2, cy + w / 2)
    ax.set_autoscale_on(False)
    return ax


def show_map(ax, img, alpha=1.0, box=None):
    if box is None:
        ext = (-1 - H / 2, 1 + H / 2, -1 - H / 2, 1 + H / 2)
    else:
        hh = box[2] / (img.shape[0] - 1)
        ext = (box[0] - box[2] / 2 - hh / 2, box[0] + box[2] / 2 + hh / 2,
               box[1] - box[2] / 2 - hh / 2, box[1] + box[2] / 2 + hh / 2)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest", alpha=alpha)


def pixel_lines(ax, view, a):
    cx, cy, w = view
    if w > 40 * H or a <= 0:
        return
    a *= 1 - ease((w - 20 * H) / (20 * H))
    e = np.concatenate([G - H / 2, [G[-1] + H / 2]])
    for v in e[(e > cx - w) & (e < cx + w)]:
        ax.axvline(v, color="#2a2e3e", lw=.9, alpha=a)
    for v in e[(e > cy - w) & (e < cy + w)]:
        ax.axhline(v, color="#2a2e3e", lw=.9, alpha=a)


def cell(ax, p, color=None, alpha=1.0, ring=0.0, ring_color="#ffffff", h=None):
    h = H if h is None else h
    x, y = p
    if color is not None and alpha > 0:
        ax.add_patch(Rectangle((x - h / 2, y - h / 2), h, h, color=color, lw=0, alpha=alpha))
    if ring > 0:
        ax.add_patch(Rectangle((x - h / 2, y - h / 2), h, h, fill=False, ec=ring_color, lw=3, alpha=ring))


def colorbar(rect, alpha=1.0, marks=()):
    if alpha <= 0.01:
        return
    ax = fig.add_axes(rect)
    ax.imshow(magma(np.linspace(0, VMAX, 256))[None, :, :3], aspect="auto", extent=(0, VMAX, 0, 1), alpha=alpha)
    ax.patch.set_alpha(0)
    ax.set_yticks([]); ax.set_xticks([0, 5, 10, 15, 20])
    ax.tick_params(colors=MUTED, labelsize=20, length=0, pad=6)
    for lab in ax.get_xticklabels():
        lab.set_alpha(alpha); lab.set_family(BODY)
    for s in ax.spines.values():
        s.set_visible(False)
    for v, col in marks:
        ax.plot([v, v], [-0.4, 1.4], color=col, lw=3.5, alpha=alpha, clip_on=False, solid_capstyle="round")
    fig.text(rect[0], rect[1] + rect[3] + 0.02, "colour = loops it took to settle", fontsize=22,
             family=BODY, color=MUTED, alpha=alpha)


def state_img(rect, zimg, alpha=1.0, label=None, size=22):
    if alpha <= 0.01:
        return
    ax = fig.add_axes(rect)
    ax.imshow(zimg, cmap=MEMCMAP, vmin=-MEMLIM, vmax=MEMLIM, aspect="auto", interpolation="nearest", alpha=alpha)
    ax.axis("off")
    if label:
        fig.text(rect[0], rect[1] + rect[3] + 0.02, label, fontsize=size, family=HEAD, color=MUTED, alpha=alpha)


def map_rgb(k):
    s = SETTLE.ravel()
    kk = min(max(k, 0), WRONGN.shape[1] - 1)
    w = WRONGN[:, kk]
    w = (w - w.min()) / max(w.max() - w.min(), 1)
    ch = CHANGED[:, kk]
    grey = (0.10 + 0.34 * w + 0.10 * ch)[:, None] * np.array([0.85, 0.92, 1.15])[None]
    return np.where((s <= k)[:, None], magma(s)[:, :3], grey).reshape(RES, RES, 3)


def bezier(p1, p2, rad, s):
    (x1, y1), (x2, y2) = p1, p2
    cx, cy = (x1 + x2) / 2 + rad * (y2 - y1), (y1 + y2) / 2 - rad * (x2 - x1)
    return ((1 - s) ** 2 * x1 + 2 * (1 - s) * s * cx + s ** 2 * x2,
            (1 - s) ** 2 * y1 + 2 * (1 - s) * s * cy + s ** 2 * y2)


def glow_dot(ax, p, size=1.0, color="#ffffff", alpha=1.0):
    if alpha <= 0.01:
        return
    for s_, a_ in ((1600, .05), (700, .10), (240, .30), (70, 1)):
        ax.scatter(*p, s=s_ * size, color=color, alpha=a_ * alpha, lw=0, zorder=5)


def slider(ov, x0, x1, y, value, name, alpha=1.0):
    if alpha <= 0.01:
        return
    ov.plot([x0, x1], [y, y], color="#343a4e", lw=6, solid_capstyle="round", alpha=alpha)
    xv = lerp(x0, x1, (value + 1) / 2)
    ov.plot([lerp(x0, x1, 0.5), xv], [y, y], color=CYAN, lw=6, solid_capstyle="round", alpha=alpha)
    ov.scatter([xv], [y], s=420, color="#ffffff", zorder=6, alpha=alpha, lw=0)
    fig.text(x0 - 0.03, y, name, fontsize=28, family=HEAD, color=INK, alpha=alpha, ha="right", va="center")
    fig.text(x1 + 0.03, y, f"{value:+.2f}", fontsize=22, family=HEAD, color=CYAN, alpha=alpha, va="center")
    for v in (-1, 0, 1):
        fig.text(lerp(x0, x1, (v + 1) / 2), y - 0.04, f"{v}", fontsize=20, family=BODY, color=DIM,
                 alpha=alpha, ha="center")


# scenes --------------------------------------------------------------------------------
FULL = [0.13, 0.1, 0.74, 0.74]
MIDRECT = [0.2, 0.2, 0.56, 0.56]
CBAR = [0.2, 0.085, 0.56, 0.028]
FULL_MAP = map_rgb(KMAX)


def sc_hook(t):
    w = 2.0 * (1 - 0.12 * ease(t / 5.0))
    ax = plane(FULL, (0.0, -0.05 * ease(t / 5), w + H))
    show_map(ax, FULL_MAP, ease(t / 0.6))
    caption(t, [(0.3, 99, ("Each pixel: the same network,", "same Sudoku, a *different start.*"))])
    kline("Here's how to read it.", 0.045, 32, t, 2.8, None, HEAD, MUTED)


def sc_puzzle(t):
    ov = overlay()
    rect = [0.29, 0.2, 0.42, 0.42]
    card(ov, rect, ease(t / 0.5))
    gv = [ease((t - 0.2 - 0.015 * i) / 0.3) for i in range(81)]
    sudoku(rect, givens=gv, fs=21, alpha=ease(t / 0.5))
    caption(t, [(0.3, 2.2, "Meet the puzzle."),
                (2.4, 99, ("And a small neural network", "trained only on *Sudoku.*"))])
    kline("No chatbot, no words.", 0.115, 26, t, 3.6, None, HEAD, MUTED)
    kline("A grid goes in, a grid comes out.", 0.07, 26, t, 3.9, None, BODY, MUTED)


LAP = 1.5


def loop_diagram(ov, t, alpha, k_now):
    rng = np.random.default_rng(k_now + 7)
    x0, y0, w, h = 0.07, 0.6, 0.36, 0.1
    card(ov, [x0, y0, w, h], alpha, glow=CYAN if (t % LAP) < 0.25 and t > 0 else None)
    ov.imshow(rng.normal(0, 1, (6, 24)), cmap=MEMCMAP, vmin=-2.2, vmax=2.2, extent=(x0, x0 + w, y0, y0 + h),
              interpolation="nearest", alpha=alpha * 0.9, aspect="auto", zorder=3)
    fig.text(x0, y0 + h + 0.03, "its memory: 49,664 numbers", fontsize=23, family=HEAD, color=MUTED, alpha=alpha)
    nx, ny, nw, nh = 0.6, 0.6, 0.3, 0.1
    ov.add_patch(FancyBboxPatch((nx, ny), nw, nh, boxstyle="round,pad=0,rounding_size=0.03",
                                fc="#1a2440", ec=CYAN, lw=2, alpha=alpha, zorder=3))
    fig.text(nx + nw / 2, ny + nh / 2, "the network", fontsize=26, family=HEAD, color=INK, alpha=alpha,
             ha="center", va="center")
    px, py, ps = nx + 0.02, 0.735, 0.07
    ov.add_patch(Rectangle((px, py), ps, ps, fill=False, ec=MUTED, lw=1.6, alpha=alpha, zorder=3))
    for j in (1, 2):
        ov.plot([px + j * ps / 3] * 2, [py, py + ps], color=MUTED, lw=1, alpha=alpha)
        ov.plot([px, px + ps], [py + j * ps / 3] * 2, color=MUTED, lw=1, alpha=alpha)
    ov.add_patch(FancyArrowPatch((px + ps / 2, py - 0.004), (px + ps / 2, ny + nh + 0.004), arrowstyle="-|>",
                                 mutation_scale=18, lw=2, color=MUTED, alpha=alpha, zorder=4))
    fig.text(px + ps + 0.02, py + ps / 2, "+ the puzzle,\n   every loop", fontsize=22, family=BODY, color=MUTED,
             alpha=alpha, va="center", linespacing=1.15)
    a1, b1 = (x0 + w + 0.01, y0 + h * 0.75), (nx - 0.01, ny + nh * 0.75)
    a2, b2 = (nx - 0.01, ny + nh * 0.25), (x0 + w + 0.01, y0 + h * 0.25)
    for p, q in ((a1, b1), (a2, b2)):
        ov.add_patch(FancyArrowPatch(p, q, connectionstyle="arc3,rad=-0.45", arrowstyle="-|>",
                                     mutation_scale=22, lw=2.4, color=MUTED, alpha=alpha, zorder=4))
    if t > 0:
        s = (t % LAP) / LAP
        pos = bezier(a1, b1, -0.45, s * 2) if s < 0.5 else bezier(a2, b2, -0.45, s * 2 - 1)
        glow_dot(ov, pos, 0.7, CYAN, alpha)
    ov.add_patch(FancyArrowPatch((x0 + w * 0.5, y0 - 0.015), (0.33, 0.45), arrowstyle="-|>",
                                 mutation_scale=22, lw=2.4, color=MUTED, alpha=alpha, zorder=4,
                                 linestyle=(0, (4, 3))))
    fig.text(0.05, 0.47, "read out\nas a guess", fontsize=24, family=BODY, color=MUTED, alpha=alpha,
             va="center", linespacing=1.2)
    fig.text(nx + nw, ny - 0.03, "(diagram)", fontsize=20, family=BODY, color=DIM, alpha=alpha, ha="right")


def sc_loop(t):
    ov = overlay()
    a = ease(t / 0.6)
    tl = max(t - 1.0, 0)
    k = min(int(tl / LAP), 7) if t > 1.0 else -1
    loop_diagram(ov, tl if t > 1.0 else 0, a, max(k, 0))
    rect = [0.33, 0.06, 0.36, 0.36]
    card(ov, rect, a)
    run_view(rect, T1, S1, k, tl / LAP, a, 16)
    counter(0.75, 0.24, k, a)
    legend(ov, 0.05, 0.3, a * ease((t - 1.4) / 0.4))
    caption(t, [(0.2, 3.4, "It thinks in loops."),
                (3.6, 8.0, ("Each loop, the network takes the", "puzzle and updates its *memory.*")),
                (8.2, 99, ("After each loop we read the", "memory out as a *guess.*"))])


STEP1 = 0.52


def sc_settle(t):
    ov = overlay()
    rect = [0.27, 0.29, 0.46, 0.46]
    card(ov, rect)
    t0 = 1.6
    kf = max(t - t0, 0) / STEP1
    k = min(int(kf), S1 + 3)
    run_view(rect, T1, S1, k, kf, 1.0, 20)
    pills(ov, 0.15, 0.15, 0.7, T1, k, S1 + 4, ease(t / 0.6))
    counter(0.06, 0.62, k)
    legend(ov, 0.775, 0.7, 1.0, changed=True, given=True)
    if k < S1:
        fig.text(0.5, 0.245, f"{int((T1[k] != SOL).sum())} cells wrong", fontsize=25, family=HEAD,
                 color="#ffb394", ha="center", va="center")
    else:
        fig.text(0.5, 0.245, f"solved, settled at loop {S1}", fontsize=26, family=HEAD, color=CYAN,
                 ha="center", va="center", alpha=ease((kf - S1) / 0.4))
    tsettle = t0 + S1 * STEP1
    caption(t, [(0.2, 2.6, ("The same run, slowed down.", f"First guess: *{int((T1[0] != SOL).sum())} cells* wrong.")),
                (2.8, tsettle - 0.3, ("It keeps rewriting the _orange_ cells,", "sometimes for the worse...")),
                (tsettle, 99, ("...then the whole grid snaps", f"into place: *solved,* loop *{S1}.*"))])


START_BIG = [0.1, 0.42, 0.8, 0.14]
START_SMALL = [0.06, 0.12, 0.42, 0.075]
MAPRECT = [0.6, 0.12, 0.35, 0.35]
T_MANY, T_DIALS, T_AXES, T_LAND = 4.6, 9.8, 14.4, 17.4
_rng = np.random.default_rng(3)
DOTS = [tuple(x) for x in _rng.uniform(-0.85, 0.85, (7, 2))]
DOT_DT = 0.65


def path(x):
    return 0.62 * np.sin(0.8 * x + 0.3), 0.58 * np.sin(1.1 * x + 1.9)


def dial_values(t):
    if t < T_MANY + 0.4:
        return P1
    if t < T_DIALS:
        i = min(int((t - T_MANY - 0.4) / DOT_DT), len(DOTS) - 1)
        return DOTS[i]
    x = t - T_DIALS
    last = DOTS[-1]
    pa, pb = path(x)
    p0a, p0b = path(0)
    a = pa - p0a + last[0]
    b = pb - p0b + last[1]
    land = ease((t - T_LAND) / 1.6)
    return (lerp(np.clip(a, -0.95, 0.95), P1[0], land), lerp(np.clip(b, -0.95, 0.95), P1[1], land))


def sc_start(t):
    ov = overlay()
    move = ease((t - T_MANY) / 1.0)
    wa, wb = dial_values(t)
    state_img(lerp_rect(START_BIG, START_SMALL, move), wa * U + wb * V, ease(t / 0.6),
              "the starting memory: 49,664 numbers", lerp(26, 21, move))
    ma = ease((t - T_MANY - 0.2) / 0.6)
    if ma > 0.01:
        ax = plane(MAPRECT, (0, 0, 2 + H), ma, ticks=t > T_AXES)
        ax.axhline(0, color="#1e2230", lw=1); ax.axvline(0, color="#1e2230", lw=1)
        if t < T_DIALS:
            n = int((t - T_MANY - 0.4) / DOT_DT) + 1 if t > T_MANY + 0.4 else 0
            for i, p in enumerate(DOTS[:n]):
                age = t - (T_MANY + 0.4 + i * DOT_DT)
                glow_dot(ax, p, 1.0 if i == n - 1 else 0.5, "#ffffff", ma * (1 if i == n - 1 else 0.45) * eout(age / 0.3))
        else:
            for p in DOTS[:-1]:
                glow_dot(ax, p, 0.5, "#ffffff", 0.45 * (1 - ease((t - T_DIALS) / 0.8)))
            glow_dot(ax, (wa, wb), 1.0, "#ffffff", ma)
        if t > T_AXES:
            aa = ease((t - T_AXES) / 0.5)
            fig.text(0.775, 0.045, "dial a", fontsize=24, family=HEAD, color=CYAN, alpha=aa, ha="center")
            fig.text(0.548, 0.295, "dial b", fontsize=24, family=HEAD, color=CYAN, alpha=aa, va="center",
                     ha="right", rotation=90)
    pa = ease((t - T_DIALS - 0.2) / 0.6)
    if pa > 0.01:
        for j, (img, name) in enumerate(((U, "fixed pattern u"), (V, "fixed pattern v"))):
            state_img([0.06 + 0.47 * j, 0.7, 0.41, 0.07], img, ease((t - T_DIALS - 0.2 - 0.3 * j) / 0.6), name, 21)
        fig.text(0.5, 0.6, r"start  =  a $\times$ u  +  b $\times$ v", fontsize=34, family=HEAD, color=INK,
                 alpha=ease((t - T_DIALS - 0.9) / 0.6), ha="center", va="center")
        da = ease((t - T_DIALS - 1.3) / 0.6)
        slider(ov, 0.11, 0.37, 0.47, wa, "a", da)
        slider(ov, 0.11, 0.37, 0.33, wb, "b", da)
    caption(t, [(0.2, T_MANY - 0.2, ("Before loop 0, the memory is set", "to a *starting value.*")),
                (T_MANY, T_DIALS - 0.2, ("Each dot on this square is a", "different *starting memory.*")),
                (T_DIALS, T_AXES - 0.2, ("A point's position sets *two dials*", "that mix two fixed patterns.")),
                (T_AXES, T_LAND + 1.4, ("Left to right turns dial *a,*", "bottom to top turns dial *b.*")),
                (T_LAND + 1.6, 99, ("This point is the start", "we just watched."))])


START_DUR = T_LAND + 4.4


def sc_colour(t):
    ov = overlay()
    rect = lerp_rect(MAPRECT, MIDRECT, ease(t / 1.2))
    ax = plane(rect, (0, 0, 2 + H), ticks=t < 1.0)
    ax.axhline(0, color="#1e2230", lw=1); ax.axvline(0, color="#1e2230", lw=1)
    fill = ease((t - 3.2) / 0.6)
    glow_dot(ax, P1, 1.2, "#ffffff", 1 - fill)
    glow_dot(ax, P1, 1.6, magma(S1), fill)
    colorbar(CBAR, ease((t - 1.2) / 0.5), [(S1, "#ffffff")] if t > 2.0 else [])
    if 2.0 < t < 4.0:
        s = ease((t - 2.2) / 1.0)
        px = MIDRECT[0] + (P1[0] + 1 + H / 2) / (2 + H) * MIDRECT[2]
        py = MIDRECT[1] + (P1[1] + 1 + H / 2) / (2 + H) * MIDRECT[3]
        bx, by = CBAR[0] + S1 / VMAX * CBAR[2], CBAR[1] + CBAR[3]
        glow_dot(ov, (lerp(bx, px, s), lerp(by, py, s)), 0.8, magma(S1), 1 - ease((t - 3.4) / 0.4))
    caption(t, [(0.2, 4.0, (f"It took *{S1} loops* to settle.", "Colour its point by that number.")),
                (4.2, 99, ("The colour is *time,*", "not right or wrong."))])


ZOOM_T = 2.4


def neighbour_view(t):
    u = ease(t / ZOOM_T)
    w = (2 + H) * ((9 * H) / (2 + H)) ** u
    return (lerp(0, MID[0], ease(t / 1.2)), lerp(0, MID[1], ease(t / 1.2)), w), u


def label_pixels(ax, alpha, texts):
    """texts: (text for P2, the left pixel; text for P1, the right one), set beside each pixel."""
    if alpha <= 0.01:
        return
    ax.text(P2[0] - 0.75 * H, P2[1], texts[0], fontsize=23, family=HEAD, color=INK, alpha=alpha,
            ha="right", va="center", bbox={**LABEL_BOX, "alpha": 0.75 * alpha})
    ax.text(P1[0] + 0.75 * H, P1[1], texts[1], fontsize=23, family=HEAD, color=INK, alpha=alpha,
            ha="left", va="center", bbox={**LABEL_BOX, "alpha": 0.75 * alpha})


STEP2 = 0.55
T_RACE = 9.0


def sc_neighbour(t):
    ov = overlay()
    if t < T_RACE:                           # part a: zoom to single pixels, show the neighbour
        a = 1 - ease((t - T_RACE + 0.5) / 0.45)
        rect = MIDRECT if t < 5.0 else lerp_rect(MIDRECT, [0.05, 0.27, 0.42, 0.42], ease((t - 5.0) / 1.0))
        view, u = neighbour_view(t)
        ax = plane(rect, view, a)
        pixel_lines(ax, view, a)
        glow_dot(ax, P1, 1.4, magma(S1), a * (1 - ease((u - 0.3) / 0.4)))
        cell(ax, P1, magma(S1), a * ease((u - 0.3) / 0.4))
        ring2 = (0.45 + 0.55 * abs(np.sin(3 * t))) * ease((t - 2.8) / 0.4)
        cell(ax, P2, None, 0, ring=a * ring2)
        label_pixels(ax, a * ease((t - 3.0) / 0.5) * (1 - ease((t - 4.8) / 0.4)), ("one pixel over", "the start we watched"))
        sa = ease((t - 5.4) / 0.6) * a
        state_img([0.54, 0.6, 0.41, 0.08], ZB, sa, "the start we watched")
        state_img([0.54, 0.42, 0.41, 0.08], ZA, sa, "one pixel over")
        if sa > 0.01:
            fig.text(0.745, 0.33, f"they differ by under {np.ceil(REL * 100):.0f}%", fontsize=26, family=HEAD,
                     color=CYAN, alpha=ease((t - 6.4) / 0.5) * a, ha="center")
        caption(t, [(0.2, 2.6, ("Zoom in until you can see", "single *pixels.*")),
                    (2.8, 5.2, ("Now look *one pixel* over,", "to the left.")),
                    (5.4, T_RACE, ("Its start is almost the", "*same* 49,664 numbers."))])
    if t < T_RACE:
        return
    t2 = t - T_RACE                           # part b: race the two starts side by side
    L, R = [0.05, 0.31, 0.42, 0.42], [0.53, 0.31, 0.42, 0.42]   # left: one pixel over, as on the map
    end = 1.0 + (S1 + 2) * STEP2 + 2.4
    a = ease((t2 - 0.05) / 0.5) * (1 - ease((t2 - end) / 0.6))
    t0 = 1.0
    kf = max(t2 - t0, 0) / STEP2
    k = min(int(kf), S1 + 2) if t2 >= t0 else -1
    for rect, tr, s, name in ((L, T2, S2, "one pixel over"), (R, T1, S1, "the start we watched")):
        card(ov, rect, a, glow=CYAN if (k >= s and kf - s < 1.5) else None)
        run_view(rect, tr, s, k, kf, a, 18)
        fig.text(rect[0] + rect[2] / 2, rect[1] + rect[3] + 0.04, name, fontsize=25, family=HEAD,
                 color=MUTED, alpha=a, ha="center")
        if k >= 0:
            txt, col = ((f"{int((tr[k] != SOL).sum())} wrong", "#ffb394") if k < s
                        else (f"solved at loop {s}", CYAN))
            fig.text(rect[0] + rect[2] / 2, rect[1] - 0.05, txt, fontsize=27, family=HEAD, color=col,
                     alpha=a, ha="center")
    counter(0.5, 0.12, k, a, "center")
    caption(t2, [(0.3, t0 + 2 * STEP2 + 0.3, ("Run both. Their first guesses", "are *identical...*")),
                 (t0 + 2 * STEP2 + 0.5, t0 + 6 * STEP2, ("...then one pixel over", f"*solves it* at loop {S2}.")),
                 (t0 + 6 * STEP2 + 0.2, t0 + S1 * STEP2 - 0.2, ("The start we watched", "keeps searching...")),
                 (t0 + S1 * STEP2, 99, (f"...until loop {S1}. Both are right.", "Only the *time* differs."))])


NEIGH_DUR = T_RACE + 1.0 + (S1 + 2) * STEP2 + 3.0


def sc_pair_colour(t):
    view, _ = neighbour_view(99)
    ax = plane(MIDRECT, view, ease(t / 0.5))
    pixel_lines(ax, view, ease(t / 0.5))
    a = ease(t / 0.5)
    cell(ax, P1, magma(S1), a, ring=a)
    cell(ax, P2, magma(S2), a * ease((t - 0.6) / 0.6), ring=a)
    label_pixels(ax, ease((t - 1.0) / 0.5), (f"{S2} loops", f"{S1} loops"))
    colorbar(CBAR, a, [(S1, "#ffffff"), (S2, "#ffffff")])
    caption(t, [(0.2, 99, ("Two neighbouring pixels,", "two very different *colours.*"))])


SLOW, STEP_ALL, FAST = 0.8, 0.4, 0.2


def all_k(x):
    """Loop index during the crystallisation: slow for loops 0-4, then faster, faster still after 12."""
    if x < 5 * SLOW:
        return x / SLOW
    x -= 5 * SLOW
    if x < 8 * STEP_ALL:
        return 5 + x / STEP_ALL
    return 13 + (x - 8 * STEP_ALL) / FAST


def sc_all(t):
    ov = overlay()
    u = ease(t / 2.4)
    rect = lerp_rect(MIDRECT, FULL, u)
    w = (9 * H) * ((2 + H) / (9 * H)) ** u
    view = (lerp(MID[0], 0, u), lerp(MID[1], 0, u), w)
    ax = plane(rect, view)
    t0 = 2.6
    if t < t0:
        k, img = -1, map_rgb(0)
    else:
        kf = all_k(t - t0)
        k = min(int(kf), KMAX)
        fr = ease((kf - int(kf) - 0.6) / 0.4) if k < KMAX else 0
        img = map_rgb(k) * (1 - fr) + map_rgb(min(k + 1, KMAX)) * fr
    if k < S1:
        img = img.copy()
        img[ROW, CB] = magma(S1)[:3]
        img[ROW, CA] = magma(S2)[:3]
    show_map(ax, img)
    pixel_lines(ax, view, 1 - u)
    ax.add_patch(Circle(MID, 2.5 * H * (1 - u) + 0.035 * u, fill=False, ec="#ffffff", lw=2.5,
                        alpha=0.9 * (1 - ease((t - 11) / 1.0))))
    if k >= 0:
        counter(0.065, 0.5, k, 1.0, "center", 40)
    la = ease((t - 0.6) / 0.5)
    chip(ov, FULL[0], 0.05, "#7a83a0", "grey: still changing", la)
    chip(ov, FULL[0] + 0.43, 0.05, magma(4), "colour: settled", la)
    caption(t, [(0.2, 2.4, ("Now do this for *every pixel:*", "40,000 starts.")),
                (2.6, 99, ("All of them think at once,", "loop by *loop.*"))])


ALL_DUR = 2.6 + 5 * SLOW + 8 * STEP_ALL + (KMAX - 12) * FAST + 1.4


def sc_read(t):
    ax = plane(FULL, (0, 0, 2 + H))
    show_map(ax, FULL_MAP)
    for p, rad, t0, col in ((SMOOTH_AT, 0.16, 0.4, CYAN), (TANGLE_AT, 0.13, 3.0, "#ffffff"), (MID, 0.06, 5.8, "#ffd166")):
        a = window(t, t0)
        if a > 0.01:
            ax.add_patch(Circle(p, rad * (1.3 - 0.3 * eout((t - t0) / 0.5)), fill=False, ec=col, lw=4, alpha=a))
    caption(t, [(0.2, 2.8, ("*Smooth* areas: neighbours", "think about as long.")),
                (3.0, 5.6, ("Tangled bands: neighbours", "*disagree.*")),
                (5.8, 99, ("The yellow ring marks our", "two starts, in a tangled band."))])


def sc_edge(t):
    """A drawing: an ordinary smooth edge straightens out under a zoom."""
    ax = plane(FULL, (0, 0, 2.2))
    ax.set_facecolor("#140e22")
    ax.add_patch(Circle((0, -0.1), 0.72, color=magma(4), lw=0))
    u = ease((t - 1.2) / 3.0)
    w = 2.2 * (0.01 / 2.2) ** u
    ex, ey = 0.0, 0.62                       # a point on the edge, kept fixed on screen as we zoom
    cx, cy = ex - ex * w / 2.2, ey - ey * w / 2.2
    ax.set_xlim(cx - w / 2, cx + w / 2)
    ax.set_ylim(cy - w / 2, cy + w / 2)
    bw = w * 0.22
    ax.add_patch(Rectangle((ex - bw / 2, ey - bw / 2), bw, bw, fill=False, ec=CYAN, lw=3,
                           alpha=0.9 * (1 - ease((u - 0.85) / 0.15))))
    fig.text(FULL[0] + FULL[2] - 0.015, FULL[1] + FULL[3] - 0.045, f"{2.2 / w:.0f}x", fontsize=30,
             family=HEAD, color=INK, ha="right", alpha=ease((t - 1.2) / 0.3))
    fig.text(0.5, 0.05, "a drawing, not data", fontsize=26, family=BODY, color=DIM, ha="center")
    caption(t, [(0.2, 99, ("Zoom into an *ordinary* smooth", "edge and it becomes a straight line."))])


ZT, HOLD, FINAL = 2.3, 0.4, 5.0


def pixel_lines_h(ax, view, h, box, a):
    """Pixel borders of a zoom level whose pixels are h wide and centred on its grid."""
    if a <= 0.01:
        return
    cx, cy, w = view
    n = int(round(box[2] / h)) + 1
    gx = np.linspace(box[0] - box[2] / 2, box[0] + box[2] / 2, n)
    gy = np.linspace(box[1] - box[2] / 2, box[1] + box[2] / 2, n)
    for v in gx[np.abs(gx - cx) < w] - h / 2:
        ax.axvline(v, color="#0d0f16", lw=1.2, alpha=a * 0.8)
    for v in gy[np.abs(gy - cy) < w] - h / 2:
        ax.axhline(v, color="#0d0f16", lw=1.2, alpha=a * 0.8)


def sc_zoom(t):
    ax = plane(FULL, (0, 0, 2))
    n = len(LEVELS) - 1
    k = min(int(t / (ZT + HOLD)), n - 1)
    u = ease((t - k * (ZT + HOLD)) / ZT)
    (f0, b0), (f1, b1) = LEVELS[k], LEVELS[k + 1]
    w = b0[2] * (b1[2] / b0[2]) ** u
    fr = (b0[2] - w) / (b0[2] - b1[2])
    cx, cy = lerp(b0[0], b1[0], fr), lerp(b0[1], b1[1], fr)
    ia = ease((u - .05) / .35)
    show_map(ax, magma(f0)[..., :3], 1 - 0.25 * ia, b0)        # coarser level dims as the finer one arrives
    if ia > 0:
        show_map(ax, magma(f1)[..., :3], ia, b1)
    ba = (1 - ease((u - .5) / .3)) * ease((t - k * (ZT + HOLD) + 0.5) / 0.4)
    if ba > 0:
        ax.add_patch(Rectangle((b1[0] - b1[2] / 2, b1[1] - b1[2] / 2), b1[2], b1[2], fill=False, ec=CYAN, lw=3, alpha=ba))
    end = n * (ZT + HOLD)
    zoom = 2.0 / w
    label = f"{zoom:.0f}x"
    if t > end:                               # enlarge the deepest level's own pixels around the pair
        e = ease((t - end) / 1.4)
        pc = ((DEEP_PAIR[0][0][0] + DEEP_PAIR[1][0][0]) / 2, DEEP_PAIR[0][0][1])
        w = w * ((12 * DEEP_H) / w) ** e
        cx, cy = lerp(cx, pc[0], e), lerp(cy, pc[1], e)
        label = f"{DEEPEST:.0f}x, single pixels" if e > 0.5 else label
        pixel_lines_h(ax, (cx, cy, w), DEEP_H, LEVELS[-1][1], ease((e - 0.6) / 0.4))
    ax.set_xlim(cx - w / 2, cx + w / 2); ax.set_ylim(cy - w / 2, cy + w / 2)
    fig.text(FULL[0] + FULL[2] - 0.015, FULL[1] + FULL[3] - 0.045, label, fontsize=30,
             family=HEAD, color=INK, ha="right", alpha=ease(t / 0.3) if zoom > 1.05 else 0)
    pa = ease((t - end - 1.2) / 0.5)
    if pa > 0:
        (p1, s1), (p2, s2) = DEEP_PAIR
        cell(ax, p1, None, 0, ring=pa, h=DEEP_H)
        cell(ax, p2, None, 0, ring=pa, h=DEEP_H)
        lo, hi = ((p1, s1), (p2, s2)) if p1[0] < p2[0] else ((p2, s2), (p1, s1))
        ax.text(lo[0][0] - 0.8 * DEEP_H, lo[0][1], f"{lo[1]} loops", fontsize=26, family=HEAD, color=INK,
                alpha=pa, ha="right", va="center", bbox={**LABEL_BOX, "alpha": 0.75 * pa})
        ax.text(hi[0][0] + 0.8 * DEEP_H, hi[0][1], f"{hi[1]} loops", fontsize=26, family=HEAD, color=INK,
                alpha=pa, ha="left", va="center", bbox={**LABEL_BOX, "alpha": 0.75 * pa})
    caption(t, [(0.0, end + 1.0, ("This one *doesn't.* Zooming into", "a tangled band...")),
                (end + 1.2, 99, (f"...at *{DEEPEST:.0f}x,* two more neighbours:", f"*{min(DEEP_PAIR[0][1], DEEP_PAIR[1][1])}* loops and *{max(DEEP_PAIR[0][1], DEEP_PAIR[1][1])}.*"))])


ZOOM_DUR = (len(LEVELS) - 1) * (ZT + HOLD) + FINAL


def sc_end(t):
    ax = plane([0, 0, 1, 1], (0, 0, 2 + H), 0.0)
    for s in ax.spines.values():
        s.set_visible(False)
    show_map(ax, FULL_MAP, 0.16)
    beats = [
        (0.2, 4.2, [("That's a *fractal* boundary:", 42, HEAD, INK), ("fast and slow starts stay mixed", 32, BODY, MUTED),
                    ("at every zoom we tried.", 32, BODY, MUTED)]),
        (4.6, 8.6, [("So how long it thinks can hinge", 36, HEAD, INK), ("on a *tiny* change in where it starts.", 36, HEAD, INK)]),
        (9.0, 13.2, [("The practical upshot:", 30, BODY, MUTED), ("if its guess keeps changing,", 40, HEAD, INK),
                     ("*restart* from another point.", 40, HEAD, INK), ("On these puzzles, that beats waiting.", 26, BODY, MUTED)]),
        (13.6, 17.8, [("Why it matters beyond Sudoku:", 30, BODY, MUTED), ("thinking in loops is one way people", 34, HEAD, INK),
                      ("are trying to make *language models*", 34, HEAD, INK), ("think longer.", 34, HEAD, INK),
                      ("Whether they behave like this is still open.", 26, BODY, MUTED)]),
    ]
    for t0, t1, lines in beats:
        y = 0.68
        for j, (s, size, font, col) in enumerate(lines):
            kline(s, y, size, t, t0 + 0.15 * j, t1, font, col)
            y -= 0.022 + size / 600
    kline("Reproduced from Lai, Bao, Quinn & Gilpin (2026)", 0.6, 26, t, 18.2, None, BODY, MUTED)
    kline("on their released models. Every frame is", 0.555, 26, t, 18.4, None, BODY, MUTED)
    kline("measured data, except the two diagrams.", 0.51, 26, t, 18.6, None, BODY, MUTED)
    kline("*infraphysics.net*", 0.38, 44, t, 19.2, None, HEAD)


SCENES = [
    (5.0, sc_hook), (5.8, sc_puzzle), (12.6, sc_loop), (1.6 + (S1 + 4) * STEP1 + 1.8, sc_settle),
    (START_DUR, sc_start), (7.0, sc_colour), (NEIGH_DUR, sc_neighbour), (4.0, sc_pair_colour),
    (ALL_DUR, sc_all), (8.6, sc_read), (5.6, sc_edge), (ZOOM_DUR, sc_zoom), (23.0, sc_end),
]
TOTAL = sum(dur for dur, _ in SCENES)


def frame(fn, t, tg):
    fig.clf()
    fig.figimage(BGIMG, 0, 0, zorder=-10)
    fn(t)
    p = fig.add_axes([0, 0, 1, 0.006]); p.axis("off"); p.set_xlim(0, 1); p.set_ylim(0, 1)
    p.add_patch(Rectangle((0, 0), tg / TOTAL, 1, color=CYAN, alpha=0.9, lw=0))
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def frames(fps=FPS):
    tg = 0.0
    for dur, fn in SCENES:
        for i in range(int(round(dur * fps))):
            yield frame(fn, i / fps, tg + i / fps)
        tg += dur


def stamps(step):
    out, tg = [], 0.0
    for dur, fn in SCENES:
        t = 0.5
        while t < dur:
            out.append((frame(fn, t, tg + t), tg + t))
            t += step
        tg += dur
    return out


if __name__ == "__main__":
    print(f"total {TOTAL:.1f} s; starts differ by {REL:.4f}; deep pair {DEEP_PAIR}")
    if MODE in ("sheet", "pages"):
        from PIL import Image, ImageDraw
        step = float(sys.argv[6]) if len(sys.argv) > 6 else 1.5
        st = stamps(step)
        if MODE == "pages":
            for pg in range(0, len(st), 9):
                page = Image.new("RGB", (1800, 1800), (0, 0, 0))
                for i, (fr, tt) in enumerate(st[pg:pg + 9]):
                    im = Image.fromarray(fr).resize((600, 600))
                    ImageDraw.Draw(im).text((8, 8), f"{tt:5.1f}s", fill=(255, 255, 0))
                    page.paste(im, (600 * (i % 3), 600 * (i // 3)))
                page.save(f"{OUT}_{pg // 9:02d}.png")
            print("wrote", (len(st) + 8) // 9, "pages")
        else:
            cols = 6
            sheet = Image.new("RGB", (360 * cols, 360 * ((len(st) + cols - 1) // cols)), (7, 8, 12))
            for i, (fr, tt) in enumerate(st):
                sheet.paste(Image.fromarray(fr).resize((360, 360)), (360 * (i % cols), 360 * (i // cols)))
            sheet.save(OUT)
            print("wrote", OUT, len(st), "frames")
        sys.exit()
    fps = 12 if MODE == "preview" else FPS
    w = imageio_ffmpeg.write_frames(OUT, (SIZE, SIZE), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                    quality=9, macro_block_size=8)
    w.send(None)
    n = 0
    for fr in frames(fps):
        w.send(fr)
        n += 1
    w.close()
    print(f"wrote {OUT}: {n} frames, {n / fps:.1f} s")
