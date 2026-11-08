#!/bin/bash
# Score each tuned model once its results/<m>_tuned.json exists. Batches whatever is
# ready so the data load and the oracle pass are shared; score_risk skips sources
# already scored, so a rerun of this loop resumes.
cd ~/MTP
while tmux has-session -t c1 2>/dev/null; do sleep 300; done   # one data load at a time (RAM)
M="dacr_amtp mha_lstm stdan inatran cognitive miat x_track traj_mamba"
while true; do
  ready=""; for m in $M; do [ -f experiments/results/${m}_tuned.json ] && ready="$ready ${m}_tuned"; done
  n=$(echo $ready | wc -w)
  if [ -n "$ready" ]; then
    out=$(OMP_NUM_THREADS=8 /usr/bin/python3 experiments/score_risk.py $ready \
          --sources cv,oracle,cv_scene,pred --sample-stride 1 \
          --out experiments/results/risk_v2 2>&1 | tee -a experiments/runs/risk_v2_tuned.out)
    [ "$n" -eq 8 ] && echo "$out" | grep -q "nothing to do" && { echo "all tuned models scored"; break; }
  fi
  sleep 600
done
