"""E3. When does the fractal appear? Basins of the GF(3) linear-system looped transformer
across its own training.

For each checkpoint (ours, from a shortened 60k-step run of loopscape's recipe) and for the
released fully trained model, draw the settle-time field on one fixed instance (the
notebook's: rng seed 10) with loopscape's settle_field, and score it.

usage: python e3_linsys_training.py RUN_DIR RES LOOPS OUT
"""
import glob
import os
import re
import sys
import time

import numpy as np
import torch

from loopscape.linsys import basin, task
from loopscape.metrics import basin_entropy, uncertainty_exponent

run_dir, RES, LOOPS, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
DEV = "cuda"

A, b, x = task.sample_instance(np.random.default_rng(10))
correct = task.answer_code(x)
print("needs elimination:", int((task.peel_depths(A) < 0).sum()), "of", len(x), "unknowns", flush=True)

ckpts = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")),
               key=lambda p: int(re.search(r"model_(\d+)", p).group(1)))
targets = [(int(re.search(r"model_(\d+)", p).group(1)), p) for p in ckpts] + [(-1, None)]  # -1 = released

out = {}
for step, path in targets:
    t0 = time.time()
    if path is None:
        model = basin.load(device=DEV)
    else:
        model = basin.load(device="cpu")          # right architecture/config ...
        sd = torch.load(path, map_location="cpu")
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)                 # ... with our weights
        model = model.to(DEV)
    model.eval()
    answers, settle = basin.settle_field(model, A, b, res=RES, half_width=2.0, loops=LOOPS)
    acc = float((answers == correct).mean())
    s = np.clip(settle, 1, None)
    sb = basin_entropy(s, box_size=5)
    ue = uncertainty_exponent(s, box_size=5)
    tag = "released" if step < 0 else f"step{step}"
    print(f"{tag:10s} {time.time() - t0:5.1f}s correct={acc:.3f} settle {settle.min()}..{settle.max()} "
          f"median {int(np.median(settle))}  Sb={sb['basin_entropy']:.3f} "
          f"Sbb={sb['boundary_basin_entropy']:.3f} alpha={ue['uncertainty_exponent']:.3f}", flush=True)
    out[f"{tag}_settle"] = settle
    out[f"{tag}_correct"] = answers == correct
    out[f"{tag}_metrics"] = np.array([acc, sb["basin_entropy"], sb["boundary_basin_entropy"],
                                      ue["uncertainty_exponent"]])
    np.savez_compressed(OUT, **out)

print("saved", OUT)
