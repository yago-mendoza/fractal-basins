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

## Running

Everything runs on a rented GPU (RunPod, RTX A5000). `tools/push.sh` and `tools/pull.sh`
move files over the RunPod SSH proxy; `tools/pod.sh` runs a command there.

```bash
pip install "loopscape[demos] @ git+https://github.com/GilpinLab/loopscape"
cd exp && python e1_sudoku_slices.py 200 24 e1_medium_200.npz medium
```
