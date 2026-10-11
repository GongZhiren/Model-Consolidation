#!/usr/bin/env bash
# Table 3: high-accuracy end-to-end endpoints.
set -euo pipefail; source "$(dirname "$0")/env.sh"
E2E="method=e2e_merge/clip_e2e_merge method.match=proj method.lr=5.0e-6 method.init_scale=0.3"
# B/32 multi-seed (8k) + representative 12k
for s in 0 1 2 3; do
  fusion_bench $E2E method.steps=8000 method.seed=$s $MP_B32 $TP_B32 \
    report_save_path="$OUTPUT_DIR/b32_8k_seed$s.json"
done
fusion_bench $E2E method.steps=12000 $MP_B32 $TP_B32 report_save_path="$OUTPUT_DIR/b32_12k.json"
# L/14 (8k)
fusion_bench $E2E method.steps=8000 $MP_L14 $TP_L14 report_save_path="$OUTPUT_DIR/l14_8k.json"
