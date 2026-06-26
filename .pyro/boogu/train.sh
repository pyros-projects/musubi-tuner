#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DIT="${DIT:-/home/pyro/models/comfy/diffusion_models/boogu_image_base_bf16.safetensors}"
VAE="${VAE:-/home/pyro/models/comfy/vae/ae.safetensors}"
TENC="${TENC:-/home/pyro/models/comfy/text_encoders/qwen3vl_8b_fp8_scaled.safetensors}"
PROCESSOR="${PROCESSOR:-/home/pyro/models/qwen3-vl-4b}"
TEXT_ENCODER_SUBFOLDER="${TEXT_ENCODER_SUBFOLDER:-auto}"
TURBO_LORA="${TURBO_LORA:-/home/pyro/models/comfy/loras/boogu/boogu_image_turbo_lora_rank_128_bf16.safetensors}"
USE_TURBO_LORA="${USE_TURBO_LORA:-auto}"  # auto = on for SMOKE, off otherwise

REQUESTED_NAME="${BOOGU_NAME:-}"
if [[ -z "$REQUESTED_NAME" ]]; then
    if [[ -n "${NAME:-}" && -f ".pyro/boogu/cfg/${NAME}.toml" && -f ".pyro/boogu/cfg/p_${NAME}.toml" ]]; then
        REQUESTED_NAME="$NAME"
    else
        if [[ -n "${NAME:-}" && "${NAME}" != "smoke" ]]; then
            echo "Ignoring inherited NAME=$NAME; set BOOGU_NAME=... to select a Boogu config." >&2
        fi
        REQUESTED_NAME="smoke"
    fi
fi
NAME="$REQUESTED_NAME"
CACHE_DATASET="${CACHE_DATASET:-0}"
SMOKE="${SMOKE:-auto}"  # auto = one-step when NAME/BOOGU_NAME resolves to smoke
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
SAMPLE_EVERY="${SAMPLE_EVERY:-50}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-100}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NETWORK_DIM="${NETWORK_DIM:-16}"
NETWORK_ALPHA="${NETWORK_ALPHA:-16}"
CACHE_LATENTS_BATCH_SIZE="${CACHE_LATENTS_BATCH_SIZE:-1}"
CACHE_TEXT_BATCH_SIZE="${CACHE_TEXT_BATCH_SIZE:-1}"

truthy() {
    case "$1" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        0|false|FALSE|no|NO|off|OFF) return 1 ;;
        *) echo "Expected boolean-ish value, got: $1" >&2; exit 1 ;;
    esac
}

if [[ "$SMOKE" == "auto" ]]; then
    if [[ "$NAME" == "smoke" ]]; then
        SMOKE=1
    else
        SMOKE=0
    fi
fi

if truthy "$SMOKE"; then
    MAX_STEPS=1
    SAVE_EVERY=1
    SAMPLE_EVERY=1
fi

if [[ "$USE_TURBO_LORA" == "auto" ]]; then
    if truthy "$SMOKE"; then
        USE_TURBO_LORA=1
    else
        USE_TURBO_LORA=0
    fi
fi

OPT_ADAMW8BIT=(
    --optimizer_type adamw8bit
    --learning_rate 1e-4
)

OPT_ADAFACTOR=(
    --optimizer_type adafactor
    --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False"
    --lr_scheduler constant
    --learning_rate 1e-4
    --max_grad_norm 0
)

OPT_PRODIGY=(
    --optimizer_type ProdigyPlusScheduleFree
    --learning_rate 1.0
    --optimizer_args "betas=(0.9,0.99)" "weight_decay=0.0"
    --max_grad_norm 0
)

case "$OPTIMIZER" in
    adamw8bit) OPT_ARGS=("${OPT_ADAMW8BIT[@]}") ;;
    adafactor) OPT_ARGS=("${OPT_ADAFACTOR[@]}") ;;
    prodigy) OPT_ARGS=("${OPT_PRODIGY[@]}") ;;
    *) echo "Unknown OPTIMIZER: $OPTIMIZER" >&2; exit 1 ;;
esac

DATASET_TOML=".pyro/boogu/cfg/${NAME}.toml"
PROMPT_TOML=".pyro/boogu/cfg/p_${NAME}.toml"
RUN_NAME="${NAME}-${OPTIMIZER}"

for path in "$DIT" "$VAE" "$DATASET_TOML" "$PROMPT_TOML"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

if [[ ! -d "$TENC" && ! -f "$TENC" ]]; then
    echo "Missing text encoder path: $TENC" >&2
    exit 1
fi

if [[ -n "$PROCESSOR" && ! -d "$PROCESSOR" && ! -f "$PROCESSOR" ]]; then
    echo "Missing processor path: $PROCESSOR" >&2
    exit 1
fi

TEXT_ENCODER_ARGS=(--text_encoder "$TENC")
if [[ -n "$PROCESSOR" ]]; then
    TEXT_ENCODER_ARGS+=(--processor "$PROCESSOR")
fi
if [[ "$TEXT_ENCODER_SUBFOLDER" != "auto" && -n "$TEXT_ENCODER_SUBFOLDER" ]]; then
    TEXT_ENCODER_ARGS+=(--text_encoder_subfolder "$TEXT_ENCODER_SUBFOLDER")
fi

SAMPLING_LORA_ARGS=()
if truthy "$USE_TURBO_LORA"; then
    if [[ ! -f "$TURBO_LORA" ]]; then
        echo "USE_TURBO_LORA=1 but missing TURBO_LORA: $TURBO_LORA" >&2
        exit 1
    fi
    SAMPLING_LORA_ARGS=(--sampling_lora_weight "$TURBO_LORA" --sampling_lora_multiplier 1.0)
fi

source .venv/bin/activate

echo "Boogu training run: dataset=$NAME output=$RUN_NAME max_steps=$MAX_STEPS"
echo "Boogu cache step: CACHE_DATASET=$CACHE_DATASET"
echo "Boogu turbo sampling LoRA: USE_TURBO_LORA=$USE_TURBO_LORA"
echo "Boogu text encoder: $TENC"
echo "Boogu processor: $PROCESSOR"

if truthy "$CACHE_DATASET"; then
    python src/musubi_tuner/boogu_image_cache_latents.py \
        --dataset_config "$DATASET_TOML" \
        --vae "$VAE" \
        --batch_size "$CACHE_LATENTS_BATCH_SIZE"

    python src/musubi_tuner/boogu_image_cache_text_encoder_outputs.py \
        --dataset_config "$DATASET_TOML" \
        "${TEXT_ENCODER_ARGS[@]}" \
        --batch_size "$CACHE_TEXT_BATCH_SIZE"
fi

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/boogu_image_train_network.py \
    --dit "$DIT" \
    --vae "$VAE" \
    "${TEXT_ENCODER_ARGS[@]}" \
    --dataset_config "$DATASET_TOML" \
    --sample_prompts "$PROMPT_TOML" \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 3.0 \
    "${OPT_ARGS[@]}" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_boogu_image --network_dim "$NETWORK_DIM" --network_alpha "$NETWORK_ALPHA" \
    --seed 42 \
    --save_every_n_steps "$SAVE_EVERY" --max_train_steps "$MAX_STEPS" \
    --save_state --save_last_n_steps_state 2 --autoresume \
    --output_dir /home/pyro/models/_out/boogu/"$RUN_NAME" \
    --output_name "$RUN_NAME" \
    --logging_dir /home/pyro/models/_out/boogu/.logs \
    --fp8_base --fp8_scaled \
    "${SAMPLING_LORA_ARGS[@]}" \
    --sample_at_first --sample_every_n_steps "$SAMPLE_EVERY" --log_with trackio
