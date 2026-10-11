#!/usr/bin/env bash
# Preliminary Flan-T5 GLUE probe (different functional readouts; see paper caveat).
set -euo pipefail; source "$(dirname "$0")/env.sh"
GLUE="modelpool=Seq2SeqLMPool/flan-t5-base_glue taskpool=flan_t5_glue_text_generation"
fusion_bench method=e2e_merge/flan_t5_e2e method.match=logit_kl $GLUE \
  report_save_path="$OUTPUT_DIR/glue_e2e.json"
