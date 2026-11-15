"""Shared helpers for the basin experiments (follow loopscape's demos/basins.ipynb)."""
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Three Sudokus of rising difficulty.
SUDOKUS = {
    # The classic Wikipedia example: 30 givens, solvable by naked singles alone.
    "easy": "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79",
    # loopscape README quickstart puzzle (sudoku_medium.png in their assets).
    "medium": ".....6.....7.3.4..5..8.....9.8.7.3...1.....9...498.7....2....4387....2......2....",
    # Sudoku-Extreme, rating 125 (loopscape notebook's FPRM example).
    "hard": "..4.6...16..3...8.....24.....9..2..4.5.......4...3.26.....8...........172....384.",
}


def mesh(res, center=(0.0, 0.0), width=2.0):
    """res x res grid of (a, b) points in a square of side `width` around `center`."""
    cx, cy = center
    g = torch.linspace(-width / 2, width / 2, res)
    x, y = torch.meshgrid(g + cx, g + cy, indexing="xy")
    return torch.stack([x.ravel(), y.ravel()], -1)


def plane(shape, seed=0):
    """Random orthonormal 2-plane through the full latent space (QR of a Gaussian)."""
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(int(np.prod(shape)), 2))
    return q[:, 0].reshape(shape), q[:, 1].reshape(shape)


def to_traces(result):
    """[steps][B] decoded grids -> (B, steps, 9, 9) int8."""
    return torch.swapaxes(torch.tensor(np.asarray(result.intermediates), dtype=torch.int8), 0, 1)


def settle_times(traces):
    """Loops until the decoded grid stops changing for good."""
    same = (traces == traces[:, -1:]).all(-1).all(-1)
    # count trailing run of True from the end: steps - (last index where it differed + 1)
    steps = same.shape[1]
    differ = ~same
    idx = torch.arange(steps).expand_as(differ)
    last_diff = torch.where(differ, idx, torch.full_like(idx, -1)).max(1).values
    return (last_diff + 1).numpy()
