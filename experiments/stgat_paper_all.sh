#!/bin/bash
# Paper-faithful STGAT-EDGRU end to end on the GPU box (see experiments/stgat_paper.py).
# Each step rewrites its own outputs, so a rerun after a failure just repeats it.
set -e
cd ~/MTP
S="/usr/bin/python3 experiments/stgat_paper.py"
$S --prepare
for a in full m1 m2 m3; do $S --train --ablation "$a"; done
$S --eval
$S --risk
$S --case
