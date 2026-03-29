#!/usr/bin/env bash

set -euo pipefail

# User-tunable values
CHECKPOINT="/abs/path/to/ltx-2.3-22b-dev-nf4.safetensors"
GEMMA_ROOT="/abs/path/to/gemma-3-12b-it-qat-q4_0-unquantized"
SPATIAL_UPSAMPLER="/abs/path/to/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DISTILLED_LORA="/abs/path/to/ltx-2.3-22b-distilled-lora.safetensors"

PROMPT="cinematic close-up of a woman walking through neon rain at night, shallow depth of field, realistic lighting"
NEGATIVE_PROMPT=""
WIDTH=1280
HEIGHT=832
FRAME_COUNT=49
FRAME_RATE=24

OUTPUT_DIR="/abs/path/to/output/ltx23_dual_stage_manual"
OUTPUT_NAME="ltx23_nf4_dualstage_manual"

# Optional concept LoRA. Leave CONCEPT_LORA empty to disable it.
CONCEPT_LORA=""
CONCEPT_LORA_MULTIPLIER="0.8"

source .venv/bin/activate

EXTRA_LORA_ARGS=()
if [[ -n "$CONCEPT_LORA" ]]; then
  EXTRA_LORA_ARGS+=(--lora_weight "$CONCEPT_LORA" --lora_multiplier "$CONCEPT_LORA_MULTIPLIER")
fi

PYTHONPATH=src python -m musubi_tuner.ltx2_generate_video \
  --device cuda \
  --mixed_precision bf16 \
  --ltx2_checkpoint "$CHECKPOINT" \
  --vae "$CHECKPOINT" \
  --ltx_version 2.3 \
  --ltx_version_check_mode error \
  --nf4_base \
  --gemma_root "$GEMMA_ROOT" \
  --gemma_load_in_4bit \
  --prompt "$PROMPT" \
  --negative_prompt "$NEGATIVE_PROMPT" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --frame_count "$FRAME_COUNT" \
  --frame_rate "$FRAME_RATE" \
  --sample_sigmas "1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0" \
  --guidance_scale 1.0 \
  --cfg_scale 1.0 \
  --sample_two_stage \
  --sample_stage1_use_distilled_lora \
  --spatial_upsampler_path "$SPATIAL_UPSAMPLER" \
  --distilled_lora_path "$DISTILLED_LORA" \
  --sample_stage2_steps 3 \
  --sample_with_offloading \
  --sample_tiled_vae \
  --sample_vae_tile_size 512 \
  --sample_vae_tile_overlap 64 \
  --output_dir "$OUTPUT_DIR" \
  --output_name "$OUTPUT_NAME" \
  "${EXTRA_LORA_ARGS[@]}"
