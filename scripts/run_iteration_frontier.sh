#!/usr/bin/env bash
# Fig. 1 (middle) / Table 4: accuracy vs. student updates on ViT-B/32.
set -euo pipefail; source "$(dirname "$0")/env.sh"
for K in 4000 8000 12000; do
  fusion_bench method=e2e_merge/clip_e2e_merge method.match=proj method.lr=5.0e-6 \
    method.steps=$K $MP_B32 $TP_B32 report_save_path="$OUTPUT_DIR/b32_steps_$K.json"
done
