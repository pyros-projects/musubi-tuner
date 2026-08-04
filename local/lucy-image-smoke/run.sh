#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/home/pyro/repos/ltx-musubi/minimax-h3-musubi
RUNTIME_DIR=/home/pyro/datasets/h3-lucy-smoke
CONFIG="$REPO_DIR/local/lucy-image-smoke/dataset.toml"
PYTHON=/home/pyro/repos/ltx-musubi/musubi-tuner/.venv/bin/python
ACCELERATE=/home/pyro/repos/ltx-musubi/musubi-tuner/.venv/bin/accelerate
COMFYUI_DIR=/home/pyro/repos/comfy-ui
COMFY_PYTHON="$COMFYUI_DIR/.venv/bin/python"
DIT=/home/pyro/models/comfy/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
TEXT_ENCODER=/home/pyro/models/comfy/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
VAE=/home/pyro/models/comfy/vae/minimax_h3_video_vae_fp16.safetensors

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

refuse_while_comfy_runs() {
  if pgrep -f '/home/pyro/repos/comfy-ui/.venv/bin/python main.py' >/dev/null; then
    echo "ComfyUI is still running; close it before using the GPU for H3 training." >&2
    exit 1
  fi
}

preflight() {
  local image_count latent_count text_count
  image_count=$(find "$RUNTIME_DIR/images" -maxdepth 1 -type f -iname '*.jpg' | wc -l)
  latent_count=$(find "$RUNTIME_DIR/cache" -maxdepth 1 -type f -name '*_h3.safetensors' | wc -l)
  text_count=$(find "$RUNTIME_DIR/cache" -maxdepth 1 -type f -name '*_h3_te.safetensors' | wc -l)

  test -x "$PYTHON"
  test -x "$ACCELERATE"
  test -x "$COMFY_PYTHON"
  test -f "$DIT"
  test -f "$TEXT_ENCODER"
  test -f "$VAE"
  test "$image_count" -eq 5
  test "$latent_count" -eq 5
  test "$text_count" -eq 5

  "$PYTHON" - "$RUNTIME_DIR/cache" <<'PY'
import sys
from pathlib import Path

from safetensors import safe_open

for path in sorted(Path(sys.argv[1]).glob("*_h3.safetensors")):
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        assert handle.metadata()["architecture"] == "minimax_h3"
        assert any(key.startswith("latents_1x") for key in keys), (path, keys)
        assert any(key.startswith("latents_audio_2x0_") for key in keys), (path, keys)

for path in sorted(Path(sys.argv[1]).glob("*_h3_te.safetensors")):
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        assert handle.metadata()["architecture"] == "minimax_h3"
        assert any(key.startswith("varlen_h3_text_embed_") for key in keys), (path, keys)
        assert any(key.startswith("varlen_h3_token_tags_") for key in keys), (path, keys)
PY
  echo "H3 Lucy preflight passed: 5 images, 5 latent caches, 5 text caches."
}

case "${1:-preflight}" in
  cache)
    refuse_while_comfy_runs
    "$COMFY_PYTHON" -m musubi_tuner.minimax_h3_cache_text_encoder_outputs \
      --dataset_config "$CONFIG" \
      --text_encoder "$TEXT_ENCODER" \
      --comfyui_path "$COMFYUI_DIR" \
      --device cuda \
      --skip_existing
    "$PYTHON" -m musubi_tuner.minimax_h3_cache_latents \
      --dataset_config "$CONFIG" \
      --vae "$VAE" \
      --vae_dtype float16 \
      --device cuda \
      --skip_existing
    ;;
  preflight)
    preflight
    ;;
  train)
    preflight
    refuse_while_comfy_runs
    exec "$ACCELERATE" launch \
      --num_cpu_threads_per_process 1 \
      --mixed_precision bf16 \
      "$REPO_DIR/src/musubi_tuner/minimax_h3_train_network.py" \
      --dit "$DIT" \
      --dataset_config "$CONFIG" \
      --sdpa \
      --mixed_precision bf16 \
      --timestep_sampling shift \
      --discrete_flow_shift 12 \
      --video_flow_shift 12 \
      --audio_flow_shift 3 \
      --weighting_scheme none \
      --audio_loss_weight 0 \
      --gradient_checkpointing \
      --blocks_to_swap 48 \
      --network_module networks.lora \
      --network_dim 8 \
      --network_alpha 8 \
      --network_args \
        "exclude_patterns=['.*']" \
        "include_patterns=[r'.*\.attn\.(qkv_proj|out_proj)']" \
      --optimizer_type adamw8bit \
      --learning_rate 1e-4 \
      --lr_scheduler constant \
      --max_train_steps 10 \
      --save_every_n_steps 10 \
      --save_precision bf16 \
      --seed 42 \
      --max_data_loader_n_workers 1 \
      --output_dir "$RUNTIME_DIR/output" \
      --output_name h3_lucy_image_smoke_r8
    ;;
  *)
    echo "usage: $0 {cache|preflight|train}" >&2
    exit 2
    ;;
esac
