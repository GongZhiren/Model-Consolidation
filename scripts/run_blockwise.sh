#!/usr/bin/env bash
# Fig. 1 (right): memory frontier via block-wise refinement.
set -euo pipefail; source "$(dirname "$0")/env.sh"
for G in 1 2 3 4 6 12; do
  fusion_bench method=e2e_merge/clip_blockwise_e2e method.group_size=$G \
    $MP_B32 $TP_B32 report_save_path="$OUTPUT_DIR/b32_block_g$G.json"
done
