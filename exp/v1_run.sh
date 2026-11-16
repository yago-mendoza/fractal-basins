#!/bin/sh
# V1 for the video: the medium map with every grid kept, then zooms into it on the same card.
cd /workspace/fb/exp
python v1_medium_traces.py 200 24 ../out/v1_medium.npz > ../out/v1.log 2>&1 || exit 1
python -c "import numpy as np; d = np.load('../out/v1_medium.npz'); np.savez('../out/v1_start.npz', medium_settle=d['settle'])"
python e2_zoom.py medium ../out/v1_start.npz 112 24 4 4 ../out/v1_zoom.npz > ../out/v1_zoom.log 2>&1
echo ALLDONE >> ../out/v1_zoom.log
