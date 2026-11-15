# fractal-basins

Reproducing the basin-slice experiments of Lai, Bao, Quinn & Gilpin,
*Fractal basins trap latent reasoning* ([arXiv:2609.04963](https://arxiv.org/abs/2609.04963)),
on the models they released, for an InfraPhysics Bits2Bricks article.

The models and the analysis library are theirs: [GilpinLab/loopscape](https://github.com/GilpinLab/loopscape)
(installed as a dependency on the GPU machine, not vendored here). This repo holds only our
experiment scripts and figures.

## Experiments

| | script | what |
|---|---|---|
| E1 | `exp/e1_sudoku_slices.py` | EqR basin slices on an easy, a medium and a hard Sudoku: settle time, solved fraction, basin entropy, uncertainty exponent |
| E2 | `exp/e2_zoom.py` | successive zooms into the most mixed spot of an E1 field |
| E3 | `exp/e3_linsys_training.py` | basins of the GF(3) linear-system looped transformer across a shortened (60k-step) run of loopscape's training recipe, against the released model |
| check | `exp/check_determinism.py` | same starting points in a rerun, a shuffled batch and a mixed batch give identical settle times (0/256 differ): the speckle is the model, not GPU noise |

## Results

| run | grid | loops | solved | distinct final grids | settle (min/median/max) | S_b | S_bb | alpha |
|---|---|---|---|---|---|---|---|---|
| easy | 40x40 | 24 | 100% | 1 | 0/0/0 | 0 | 0 | n/a |
| medium | 200x200 | 24 | 100% | 1 | 1/3/20 | 0.619 | 1.022 | 0.308 |
| hard | 128x128 | 48 | 88.1% | 1676 | 3/22/47 | 2.737 | 2.737 | 0.000 |

E2 zooms (EqR, same plane). Medium, 128x128, 24 loops, 4x per level: widths 0.5 / 0.125 / 0.03125 give alpha 0.263 / 0.344 / 0.185 and Sbb 1.12 / 1.11 / 1.23; the speckled bands persist down to 64x. Hard, 96x96, 48 loops, 10x per level: width 0.2 shows a smooth fast-settling island (alpha 0.008), width 0.02 a clean island edge against the noise (alpha 0.025, Sb drops 2.63 to 1.55).

E3, linear systems over GF(3), 96x96, 256 loops. Our shortened run (60k of the recipe's 300k steps, RTX A5000, 2.4 h) plateaus at 39.7% of held-out systems solved from step ~20k on (`out/e3_train_log.txt`).

| model | instance | correct | settle (median) | S_b | alpha |
|---|---|---|---|---|---|
| ours, steps 2k to 60k | hard (rng 10, 8/8 unknowns need elimination) | 0% | 3 to 23 | 0.002 to 0.53 | 0.52 to 1.01 |
| released | hard | 99.9% | 40..255 (49) | 1.81 | 0.10 |
| ours, step 60k | easy (rng 0, all by substitution) | 100% | 7 | 0 | n/a (flat) |
| released | easy | 100% | 5..7 (6) | 0.06 | 0.95 |

Fractal basins show up only where the instance is hard *and* the model solves it: a model that cannot solve it settles fast and smoothly on a wrong answer, and an easy instance needs no search. Our run never reached the second skill (elimination), so it does not show the transition itself; the full recipe would (~12 h on an A5000).

## Running

Everything runs on a rented GPU (RunPod, RTX A5000). `tools/push.sh` and `tools/pull.sh`
move files over the RunPod SSH proxy; `tools/pod.sh` runs a command there.

```bash
pip install "loopscape[demos] @ git+https://github.com/GilpinLab/loopscape"
cd exp && python e1_sudoku_slices.py 200 24 e1_medium_200.npz medium
```
