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

## Results

| run | grid | loops | solved | distinct final grids | settle (min/median/max) | S_b | S_bb | alpha |
|---|---|---|---|---|---|---|---|---|
| easy | 40x40 | 24 | 100% | 1 | 0/0/0 | 0 | 0 | n/a |
| medium | 200x200 | 24 | 100% | 1 | 1/3/20 | 0.619 | 1.022 | 0.308 |
| hard | 128x128 | 48 | 88.1% | 1676 | 3/22/47 | 2.737 | 2.737 | 0.000 |

## Running

Everything runs on a rented GPU (RunPod, RTX A5000). `tools/push.sh` and `tools/pull.sh`
move files over the RunPod SSH proxy; `tools/pod.sh` runs a command there.

```bash
pip install "loopscape[demos] @ git+https://github.com/GilpinLab/loopscape"
cd exp && python e1_sudoku_slices.py 200 24 e1_medium_200.npz medium
```
