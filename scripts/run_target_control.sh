#!/usr/bin/env bash
# Table 5: target (representation vs. logit) x init, plus the SAM control.
set -euo pipefail; source "$(dirname "$0")/env.sh"
for M in proj logit; do for A in 0.0 0.3; do
  fusion_bench method=e2e_merge/clip_e2e_merge method.match=$M method.init_scale=$A \
    method.steps=8000 $MP_B32 $TP_B32 report_save_path="$OUTPUT_DIR/b32_${M}_a${A}.json"
done; done
# SAMerging-style control: logit target + sharpness-aware minimization (rho sweep)
for RHO in 0.02 0.05 0.10; do
  fusion_bench method=e2e_merge/clip_e2e_merge method.match=logit method.sam_rho=$RHO \
    method.init_scale=0.3 method.steps=6000 $MP_B32 $TP_B32 \
    report_save_path="$OUTPUT_DIR/b32_logit_sam$RHO.json"
done
