#!/usr/bin/env bash
# Fig. 1 (left): data-budget crossover on ViT-B/32 (fixed calibration set of N examples/task).
set -euo pipefail; source "$(dirname "$0")/env.sh"
for N in 64 128 256 512 1024; do
  fusion_bench method=e2e_merge/clip_e2e_merge method.match=proj method.stream=false \
    method.num_cal_batches=$((N/32)) method.steps=8000 method.lr=5.0e-6 \
    $MP_B32 $TP_B32 report_save_path="$OUTPUT_DIR/b32_fixedN_$N.json"
done
