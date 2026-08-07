#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# First run: CACHE_DATASET=1 .pyro/h3/train.sh
# Reuse cache: .pyro/h3/train.sh
# Short check: MAX_STEPS=10 SAVE_EVERY=10 .pyro/h3/train.sh
# Repeat-5 A/B: H3_NAME=lucy IMAGE_FRAME_COUNT=5 CACHE_DATASET=1 .pyro/h3/train.sh
# Silent-audio A/B: IMAGE_AUDIO_MODE=silent .pyro/h3/train.sh
# Native H3 video schedule: CACHE_DATASET=1 TIMESTEP_PRESET=h3_video IMAGE_FRAME_COUNT=5 .pyro/h3/train.sh
MUSUBI_VENV="${MUSUBI_VENV:-/home/pyro/repos/ltx-musubi/musubi-tuner/.venv}"
COMFYUI_DIR="${COMFYUI_DIR:-/home/pyro/repos/comfy-ui}"
DIT="${DIT:-/home/pyro/models/comfy/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
TEXT_ENCODER="${TEXT_ENCODER:-/home/pyro/models/comfy/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors}"
VAE="${VAE:-/home/pyro/models/comfy/vae/minimax_h3_video_vae_fp16.safetensors}"
AUDIO_VAE="${AUDIO_VAE:-/home/pyro/models/comfy/vae/minimax_h3_audio_vae_fp32.safetensors}"

H3_NAME="${H3_NAME:-cheststand}"
CACHE_DATASET="${CACHE_DATASET:-1}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-100}"
SAMPLE_EVERY="${SAMPLE_EVERY:-50}"
SAMPLE_PROMPTS="${SAMPLE_PROMPTS:-.pyro/h3/cfg/p_${H3_NAME}.toml}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NETWORK_DIM="${NETWORK_DIM:-32}"
NETWORK_ALPHA="${NETWORK_ALPHA:-$NETWORK_DIM}"
LORA_PRESET="${LORA_PRESET:-no_packed_attn}" # attn, attn_mlp, no_packed_attn, or full
OPTIMIZER="${OPTIMIZER:-prodigy}"  # adamw8bit | adafactor | prodigy
LEARNING_RATE="${LEARNING_RATE:-}"   # empty = optimizer-specific default
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-6}"
SAMPLE_BLOCKS_TO_SWAP="${SAMPLE_BLOCKS_TO_SWAP:-0}"  # 0 = unswapped snapshots, "inherit" = use BLOCKS_TO_SWAP
CACHE_LATENTS_BATCH_SIZE="${CACHE_LATENTS_BATCH_SIZE:-8}"
CACHE_TEXT_BATCH_SIZE="${CACHE_TEXT_BATCH_SIZE:-1}"
IMAGE_FRAME_COUNT="${IMAGE_FRAME_COUNT:-1}"
IMAGE_AUDIO_MODE="${IMAGE_AUDIO_MODE:-none}"
AUDIO_LOSS_WEIGHT="${AUDIO_LOSS_WEIGHT:-0}"  # set 1 for video datasets with real audio
TIMESTEP_PRESET="${TIMESTEP_PRESET:-image}"
SAMPLE_LATENT_FRAMES="${SAMPLE_LATENT_FRAMES:-2}"
SAMPLE_AUDIO_MODE="${SAMPLE_AUDIO_MODE:-auto}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-ab2}"
SAMPLE_FRAME_SELECT="${SAMPLE_FRAME_SELECT:-dup_last}"
# Optional frozen LoRA stacked on previews only (path[:strength]), e.g. the
# community Turbo LoRA for 4-step previews. Empty = off (unchanged behavior).
SAMPLE_LORA_OVERLAY="${SAMPLE_LORA_OVERLAY:-/home/pyro/models/comfy/loras/minimax/minimax_h3_turbo_4step_ckpt850.safetensors}"
SAMPLE_LORA_TEMB_GRID="${SAMPLE_LORA_TEMB_GRID:-$COMFYUI_DIR/custom_nodes/comfyui-minimax-h3-turbo/h3_silu_temb_grid.safetensors}"

if [[ ! "$H3_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "H3_NAME may only contain letters, numbers, dot, underscore, and dash: $H3_NAME" >&2
    exit 1
fi
if [[ ! "$SAMPLE_EVERY" =~ ^[0-9]+$ ]]; then
    echo "SAMPLE_EVERY must be a non-negative integer: $SAMPLE_EVERY" >&2
    exit 1
fi
if [[ "$IMAGE_FRAME_COUNT" != 1 && "$IMAGE_FRAME_COUNT" != 5 ]]; then
    echo "IMAGE_FRAME_COUNT must be 1 or 5: $IMAGE_FRAME_COUNT" >&2
    exit 1
fi
case "$IMAGE_AUDIO_MODE" in
    none|silent) ;;
    *) echo "IMAGE_AUDIO_MODE must be none or silent: $IMAGE_AUDIO_MODE" >&2; exit 1 ;;
esac
if [[ "$SAMPLE_LATENT_FRAMES" != 1 && "$SAMPLE_LATENT_FRAMES" != 2 ]]; then
    echo "SAMPLE_LATENT_FRAMES must be 1 or 2: $SAMPLE_LATENT_FRAMES" >&2
    exit 1
fi
case "$SAMPLE_AUDIO_MODE" in
    auto|none|silent) ;;
    *) echo "SAMPLE_AUDIO_MODE must be auto, none, or silent: $SAMPLE_AUDIO_MODE" >&2; exit 1 ;;
esac
case "$SAMPLE_SOLVER" in
    euler|ab2) ;;
    *) echo "SAMPLE_SOLVER must be euler or ab2: $SAMPLE_SOLVER" >&2; exit 1 ;;
esac
case "$SAMPLE_FRAME_SELECT" in
    dup_last|first|last|sharpest) ;;
    *) echo "SAMPLE_FRAME_SELECT must be dup_last, first, last, or sharpest: $SAMPLE_FRAME_SELECT" >&2; exit 1 ;;
esac
case "$LORA_PRESET" in
    attn|attn_mlp|no_packed_attn|full) ;;
    *) echo "LORA_PRESET must be attn, attn_mlp, no_packed_attn, or full: $LORA_PRESET" >&2; exit 1 ;;
esac
case "$TIMESTEP_PRESET" in
    image_v0)
        TIMESTEP_SUFFIX=""
        TIMESTEP_ARGS=(
            --timestep_sampling krea2_shift
            --discrete_flow_shift 12
        )
        ;;
    image)
        TIMESTEP_SUFFIX=""
        TIMESTEP_ARGS=(
            --timestep_sampling krea2_shift
            --discrete_flow_shift 12
            --preserve_distribution_shape
            --min_timestep 0
            --max_timestep 875
        )
        ;;
    h3_video)
        TIMESTEP_SUFFIX="-h3shift12"
        TIMESTEP_ARGS=(
            --timestep_sampling shift
            --discrete_flow_shift 12
        )
        ;;
    *) echo "TIMESTEP_PRESET must be image or h3_video: $TIMESTEP_PRESET" >&2; exit 1 ;;
esac

case "$OPTIMIZER" in
    adamw8bit)
        LEARNING_RATE="${LEARNING_RATE:-1e-4}"
        OPT_ARGS=(--optimizer_type adamw8bit --learning_rate "$LEARNING_RATE" --lr_scheduler constant)
        ;;
    adafactor)
        LEARNING_RATE="${LEARNING_RATE:-2e-4}"
        OPT_ARGS=(
            --optimizer_type adafactor
            --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False"
            --learning_rate "$LEARNING_RATE"
            --lr_scheduler constant
            --max_grad_norm 0
        )
        ;;
    prodigy)
        LEARNING_RATE="${LEARNING_RATE:-1.0}"
        OPT_ARGS=(
            --optimizer_type ProdigyPlusScheduleFree
            --optimizer_args "betas=(0.9,0.99)" "weight_decay=0.0"
            --learning_rate "$LEARNING_RATE"
            --max_grad_norm 0
        )
        ;;
    *) echo "OPTIMIZER must be adamw8bit, adafactor, or prodigy: $OPTIMIZER" >&2; exit 1 ;;
esac

truthy() {
    case "$1" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        0|false|FALSE|no|NO|off|OFF) return 1 ;;
        *) echo "CACHE_DATASET must be 0/1, true/false, yes/no, or on/off: $1" >&2; exit 1 ;;
    esac
}


refuse_while_comfy_runs() {
    local comfy_root pid process_root
    comfy_root="$(readlink -f "$COMFYUI_DIR")"
    while read -r pid; do
        process_root="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
        if [[ "$process_root" == "$comfy_root" ]]; then
            echo "ComfyUI is still running (PID $pid); close it before H3 caching or training." >&2
            exit 1
        fi
    done < <(pgrep -f '(^|/)python([0-9.]*)? main\.py([[:space:]]|$)' || true)
}

PYTHON="$MUSUBI_VENV/bin/python"
ACCELERATE="$MUSUBI_VENV/bin/accelerate"
COMFY_PYTHON="$COMFYUI_DIR/.venv/bin/python"
FRAME_SUFFIX=""
if [[ "$IMAGE_FRAME_COUNT" == 5 ]]; then
    FRAME_SUFFIX="-5f"
fi
AUDIO_SUFFIX=""
if [[ "$IMAGE_AUDIO_MODE" == silent ]]; then
    AUDIO_SUFFIX="-silentaudio"
fi
DATASET_TOML=".pyro/h3/cfg/${H3_NAME}${FRAME_SUFFIX}.toml"
LORA_TAG="$LORA_PRESET"
if [[ "$LORA_PRESET" == full ]]; then
    LORA_TAG="all"
fi
RUN_NAME="${H3_NAME}${FRAME_SUFFIX}${AUDIO_SUFFIX}${TIMESTEP_SUFFIX}-${LORA_TAG}-r${NETWORK_DIM}-${TIMESTEP_PRESET}-${IMAGE_AUDIO_MODE}-${OPTIMIZER}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/pyro/models/_out/h3/$RUN_NAME}"

for path in "$PYTHON" "$ACCELERATE"; do
    if [[ ! -x "$path" ]]; then
        echo "Missing executable: $path" >&2
        exit 1
    fi
done
for path in "$DIT" "$DATASET_TOML"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

SAMPLE_ARGS=()
if (( SAMPLE_EVERY > 0 )); then
    for path in "$VAE" "$SAMPLE_PROMPTS"; do
        if [[ ! -f "$path" ]]; then
            echo "Missing H3 preview file: $path" >&2
            exit 1
        fi
    done
    SAMPLE_ARGS=(
        --vae "$VAE"
        --sample_at_first
        --vae_dtype float16
        --sample_prompts "$SAMPLE_PROMPTS"
        --sample_every_n_steps "$SAMPLE_EVERY"
        --sample_latent_frames "$SAMPLE_LATENT_FRAMES"
        --sample_audio_mode "$SAMPLE_AUDIO_MODE"
        --sample_solver "$SAMPLE_SOLVER"
        --sample_frame_select "$SAMPLE_FRAME_SELECT"
    )
    if [[ "$SAMPLE_BLOCKS_TO_SWAP" != "inherit" ]]; then
        SAMPLE_ARGS+=(--sample_blocks_to_swap "$SAMPLE_BLOCKS_TO_SWAP")
    fi
    if [[ -n "$SAMPLE_LORA_OVERLAY" ]]; then
        SAMPLE_ARGS+=(--sample_lora_overlay "$SAMPLE_LORA_OVERLAY")
        if [[ -f "$SAMPLE_LORA_TEMB_GRID" ]]; then
            SAMPLE_ARGS+=(--sample_lora_overlay_temb_grid "$SAMPLE_LORA_TEMB_GRID")
        fi
    fi
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
refuse_while_comfy_runs

echo "H3 image LoRA: dataset=$H3_NAME output=$RUN_NAME steps=$MAX_STEPS rank=$NETWORK_DIM optimizer=$OPTIMIZER lr=$LEARNING_RATE"
echo "H3 cache step: CACHE_DATASET=$CACHE_DATASET"
echo "H3 image cache: frames=$IMAGE_FRAME_COUNT config=$DATASET_TOML"
echo "H3 image audio: mode=$IMAGE_AUDIO_MODE"
echo "H3 timesteps: preset=$TIMESTEP_PRESET"
echo "H3 LoRA targets: preset=$LORA_PRESET"
echo "H3 preview: every=$SAMPLE_EVERY prompts=$SAMPLE_PROMPTS latents=$SAMPLE_LATENT_FRAMES audio=$SAMPLE_AUDIO_MODE solver=$SAMPLE_SOLVER frame=$SAMPLE_FRAME_SELECT sample_blocks_to_swap=$SAMPLE_BLOCKS_TO_SWAP"

CACHE_DATASET_ENABLED=0
if truthy "$CACHE_DATASET"; then
    CACHE_DATASET_ENABLED=1
fi

if (( CACHE_DATASET_ENABLED || SAMPLE_EVERY > 0 )); then
    if [[ ! -x "$COMFY_PYTHON" ]]; then
        echo "Missing ComfyUI Python: $COMFY_PYTHON" >&2
        exit 1
    fi
    if [[ ! -f "$TEXT_ENCODER" ]]; then
        echo "Missing H3 text encoder: $TEXT_ENCODER" >&2
        exit 1
    fi

    TEXT_CACHE_ARGS=()
    if (( SAMPLE_EVERY > 0 )); then
        TEXT_CACHE_ARGS+=(--precache_sample_prompts --sample_prompts "$SAMPLE_PROMPTS")
    fi
    if (( ! CACHE_DATASET_ENABLED )); then
        TEXT_CACHE_ARGS+=(--cache_sample_prompts_only)
    fi

    "$COMFY_PYTHON" -m musubi_tuner.minimax_h3_cache_text_encoder_outputs \
        --dataset_config "$DATASET_TOML" \
        --text_encoder "$TEXT_ENCODER" \
        --comfyui_path "$COMFYUI_DIR" \
        --device cuda \
        --batch_size "$CACHE_TEXT_BATCH_SIZE" \
        --skip_existing \
        "${TEXT_CACHE_ARGS[@]}"
fi

if (( CACHE_DATASET_ENABLED )); then
    if [[ ! -f "$VAE" ]]; then
        echo "Missing H3 VAE: $VAE" >&2
        exit 1
    fi
    AUDIO_VAE_ARGS=()
    if [[ -f "$AUDIO_VAE" ]]; then
        AUDIO_VAE_ARGS=(--audio_vae "$AUDIO_VAE")
    fi
    "$PYTHON" -m musubi_tuner.minimax_h3_cache_latents \
        --dataset_config "$DATASET_TOML" \
        --vae "$VAE" \
        --vae_dtype float16 \
        "${AUDIO_VAE_ARGS[@]}" \
        --device cuda \
        --batch_size "$CACHE_LATENTS_BATCH_SIZE" \
        --image_frame_count "$IMAGE_FRAME_COUNT" \
        --skip_existing
fi

exec "$ACCELERATE" launch \
    --num_cpu_threads_per_process 1 \
    --mixed_precision bf16 \
    "$ROOT/src/musubi_tuner/minimax_h3_train_network.py" \
    --dit "$DIT" \
    --dataset_config "$DATASET_TOML" \
    --sdpa \
    --mixed_precision bf16 \
    "${TIMESTEP_ARGS[@]}" \
    --video_flow_shift 12 \
    --audio_flow_shift 3 \
    --weighting_scheme none \
    --audio_loss_weight "$AUDIO_LOSS_WEIGHT" \
    --image_audio_mode "$IMAGE_AUDIO_MODE" \
    --gradient_checkpointing \
    --blocks_to_swap "$BLOCKS_TO_SWAP" \
    --network_module networks.lora \
    --network_dim "$NETWORK_DIM" \
    --network_alpha "$NETWORK_ALPHA" \
    --lora_target_preset "$LORA_PRESET" \
    "${OPT_ARGS[@]}" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --max_train_steps "$MAX_STEPS" \
    --save_every_n_steps "$SAVE_EVERY" --save_state --save_last_n_steps_state 2 \
    --save_precision bf16 \
    "${SAMPLE_ARGS[@]}" \
    --seed 42 \
    --max_data_loader_n_workers 1 \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$RUN_NAME" \
    --autoresume
