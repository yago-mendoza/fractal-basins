"""E6. Where to start thinking: random starts, restarts, and a start found by evolution.

Starts live in a D-dimensional random subspace of EqR's (97, 512) latent: z = sum_i c_i e_i,
e_i orthonormal (QR of a Gaussian, seed 1), random starts c ~ N(0, 1/D) so |z| is about 1,
the scale of the E1 slices.

1. Evolve. CMA-ES over c, minimising the mean settle time (unsettled counts as MAX + 24) on
   TRAIN hard puzzles (rating >= 100, Sudoku-Extreme test split, seed 0).
2. Trace. On TEST hard puzzles (disjoint), run K random starts, the evolved start and the zero
   start for MAX loops each, keeping the decoded grid of every loop.
3. Policies are then replayed offline from those traces, with an online stopping rule only
   (answer unchanged for k loops), by e6_policies.py.

usage: python e6_starts.py CSV D N_TRAIN N_TEST K MAX GENS POP OUT
"""
import csv
import sys
import time

import numpy as np
import torch
import cma

from loopscape import get_solver

from common import DEV, to_traces, settle_times

CSV, D, N_TRAIN, N_TEST, K, MAX, GENS, POP, OUT = sys.argv[1:10]
D, N_TRAIN, N_TEST, K, MAX, GENS, POP = map(int, (D, N_TRAIN, N_TEST, K, MAX, GENS, POP))

rows = [r for r in csv.DictReader(open(CSV)) if int(r["rating"]) >= 100]
rng = np.random.default_rng(0)
pick = [rows[i] for i in rng.choice(len(rows), N_TRAIN + N_TEST, replace=False)]
train, test = pick[:N_TRAIN], pick[N_TRAIN:]
print("train ratings", [int(r["rating"]) for r in train], "test ratings median",
      np.median([int(r["rating"]) for r in test]), flush=True)

torch.manual_seed(1)
E, _ = torch.linalg.qr(torch.randn(97 * 512, D))           # (97*512, D)
E = E.T.reshape(D, 97, 512)
solver = get_solver("eqr", device=DEV)


def latents(C):
    C = torch.as_tensor(np.asarray(C), dtype=torch.float32)
    return torch.einsum("bd,dsh->bsh", C, E)


def run(puzzle, C, steps):
    z = latents(C)
    out = solver.solve_batch(puzzle, z_H=z, z_L=z * 0, max_steps=steps, noise_scale=0.0,
                             return_intermediates=True)
    return to_traces(out).numpy().reshape(len(C), steps, 81)


def cost(C):
    tot = np.zeros(len(C))
    for r in train:
        tr = run(r["question"], C, MAX)
        st = settle_times(torch.as_tensor(tr.reshape(len(C), MAX, 9, 9)))
        tot += np.where(st >= MAX - 1, MAX + 24, st)
    return tot / len(train)


# 1. evolve
es = cma.CMAEvolutionStrategy(np.zeros(D), 1 / np.sqrt(D), {"popsize": POP, "seed": 0, "verbose": -9})
hist = []
for g in range(GENS):
    t0 = time.time()
    C = es.ask()
    f = cost(C)
    es.tell(C, f.tolist())
    hist.append([f.min(), np.median(f), es.result.fbest])
    print(f"gen {g:2d} {time.time() - t0:5.1f}s best {f.min():5.1f} median {np.median(f):5.1f} "
          f"best ever {es.result.fbest:5.1f}", flush=True)
c_star = np.asarray(es.result.xbest)
rand0 = rng.normal(0, 1 / np.sqrt(D), (POP * 4, D))
print("train cost: evolved", cost(c_star[None])[0], " zero", cost(np.zeros((1, D)))[0],
      " random median", np.median(cost(rand0)), flush=True)

# 2. trace on held-out puzzles
R = rng.normal(0, 1 / np.sqrt(D), (K, D))
starts = np.concatenate([R, c_star[None], np.zeros((1, D))])   # K random, evolved, zero
traces, sols = [], []
for j, r in enumerate(test):
    t0 = time.time()
    traces.append(run(r["question"], starts, MAX).astype(np.int8))
    sols.append(np.array([int(c) for c in r["answer"]], dtype=np.int8))
    ok = (traces[-1][:, -1] == sols[-1]).all(1)
    print(f"test {j:2d} rating {int(r['rating']):4d} {time.time() - t0:5.1f}s random right {ok[:K].mean():.2f} "
          f"evolved {'right' if ok[K] else 'wrong'} zero {'right' if ok[K + 1] else 'wrong'}", flush=True)
    np.savez_compressed(OUT, traces=np.array(traces), sols=np.array(sols), c_star=c_star, R=R,
                        hist=np.array(hist), ratings=np.array([int(x["rating"]) for x in test[:j + 1]]))
print("saved", OUT)
