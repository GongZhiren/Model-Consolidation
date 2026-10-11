#!/usr/bin/env bash
# Edit for your environment, then:  source scripts/env.sh
# FusionBench checkout that contains the registered e2e_merge method + configs:
export FUSIONBENCH_ROOT="${FUSIONBENCH_ROOT:-$PWD/fusion_bench}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs}"
export GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUTPUT_DIR"

# Standard eight-task pools (override base_model for other backbones)
export MP_B32="modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8"
export TP_B32="taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 taskpool.base_model=openai/clip-vit-base-patch32"
export MP_L14="modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8"
export TP_L14="taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 taskpool.base_model=openai/clip-vit-large-patch14"
