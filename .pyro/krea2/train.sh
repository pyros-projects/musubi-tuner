#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# User-tunable values. Every value can be overridden from the environment:
#   NAME=emo SAMPLE_EVERY=25 .pyro/krea2/train.sh
#   NAME=msplits LORA_CONFIG=preset-1 .pyro/krea2/train.sh
#   NAME=msplits CACHE_DATASET=1 .pyro/krea2/train.sh
#   NAME=msplits SAMPLE_WITH_OFFLOADING=0 .pyro/krea2/train.sh
#   NAME=msplits BYPASS=/home/pyro/models/comfy/loras/krea/krea2filterbypass3.safetensors .pyro/krea2/train.sh
DIT="${DIT:-/home/pyro/models/comfy/diffusion_models/krea2_raw_bf16.safetensors}"
TENC="${TENC:-/home/pyro/models/comfy/text_encoders/qwen3vl_4b_bf16.safetensors}"
VAE="${VAE:-/home/pyro/models/comfy/vae/qwen_image_vae.safetensors}"
TURBO_LORA="${TURBO_LORA:-/home/pyro/models/comfy/loras/krea/krea2_turbo_lora_rank_64_bf16.safetensors}"
BYPASS="${BYPASS:-}"
BYPASS_WEIGHT="${BYPASS_WEIGHT:-5}"

NAME="${NAME:-cobra}"
LORA_CONFIG="${LORA_CONFIG:-default}"   # default | preset-1 | preset-2 | preset-3 | custom
NETWORK_ARGS="${NETWORK_ARGS:-}"        # optional raw Musubi --network_args entries, e.g. "verbose=True"
CACHE_DATASET="${CACHE_DATASET:-0}"     # 1/true/yes/on = cache latents and text encoder outputs before training
OPTIMIZER="${OPTIMIZER:-adamw8bit}"   # adamw8bit | adafactor | prodigy
SAMPLE_EVERY="${SAMPLE_EVERY:-50}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-100}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NETWORK_DIM="${NETWORK_DIM:-16}"
NETWORK_ALPHA="${NETWORK_ALPHA:-16}"
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-10}"
SAMPLE_BLOCKS_TO_SWAP="${SAMPLE_BLOCKS_TO_SWAP:-0}"  # 0 = unswapped snapshots, "inherit" = use BLOCKS_TO_SWAP
SAMPLE_WITH_OFFLOADING="${SAMPLE_WITH_OFFLOADING:-1}"  # 1 = offload DiT before snapshot VAE decode
CACHE_LATENTS_BATCH_SIZE="${CACHE_LATENTS_BATCH_SIZE:-2}"
CACHE_TEXT_BATCH_SIZE="${CACHE_TEXT_BATCH_SIZE:-1}"

case "$CACHE_DATASET" in
    1|true|TRUE|yes|YES|on|ON)
        CACHE_DATASET_ENABLED=1
        ;;
    0|false|FALSE|no|NO|off|OFF)
        CACHE_DATASET_ENABLED=0
        ;;
    *)
        echo "CACHE_DATASET must be 0/1, true/false, yes/no, or on/off: $CACHE_DATASET" >&2
        exit 1
        ;;
esac

case "$SAMPLE_WITH_OFFLOADING" in
    1|true|TRUE|yes|YES|on|ON)
        SAMPLE_WITH_OFFLOADING_ENABLED=1
        ;;
    0|false|FALSE|no|NO|off|OFF)
        SAMPLE_WITH_OFFLOADING_ENABLED=0
        ;;
    *)
        echo "SAMPLE_WITH_OFFLOADING must be 0/1, true/false, yes/no, or on/off: $SAMPLE_WITH_OFFLOADING" >&2
        exit 1
        ;;
esac

if [[ ! "$LORA_CONFIG" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "LORA_CONFIG may only contain letters, numbers, dot, underscore, and dash: $LORA_CONFIG" >&2
    exit 1
fi

RUN_NAME="${NAME}-${LORA_CONFIG}-${OPTIMIZER}"
LORA_PRESET_ARGS=()

case "$LORA_CONFIG" in
    default)
        # Krea2 module default: all 264 Linear layers in the DiT.
        ;;
    preset-1)
        # Main per-block attention only: wq/wk/wv/wo/gate.
        LORA_PRESET_ARGS=("include_patterns=['blocks\\.[0-9]+\\.attn\\.(wq|wk|wv|wo|gate)']")
        ;;
    preset-2)
        # Main per-block attention + MLP, excluding first/last/tproj/txtfusion extras.
        LORA_PRESET_ARGS=("include_patterns=['blocks\\.[0-9]+\\.(attn\\.(wq|wk|wv|wo|gate)|mlp\\.(gate|up|down))']")
        ;;
    preset-3)
        # Diffusion-pipe-ish: any Linear module whose path contains "blocks.".
        LORA_PRESET_ARGS=("include_patterns=['.*blocks\\..*']")
        ;;
    custom)
        if [[ -z "$NETWORK_ARGS" ]]; then
            echo "LORA_CONFIG=custom requires NETWORK_ARGS='key=value ...'." >&2
            exit 1
        fi
        ;;
    *)
        echo "Unknown LORA_CONFIG: $LORA_CONFIG" >&2
        echo "Valid values: default, preset-1, preset-2, preset-3, custom" >&2
        exit 1
        ;;
esac

NETWORK_ARGS_VALUES=("${LORA_PRESET_ARGS[@]}")
if [[ -n "$NETWORK_ARGS" ]]; then
    # NETWORK_ARGS is intentionally split into Musubi key=value entries.
    read -r -a EXTRA_NETWORK_ARGS <<< "$NETWORK_ARGS"
    NETWORK_ARGS_VALUES+=("${EXTRA_NETWORK_ARGS[@]}")
fi

NETWORK_ARGS_CLI=()
if ((${#NETWORK_ARGS_VALUES[@]} > 0)); then
    NETWORK_ARGS_CLI=(--network_args "${NETWORK_ARGS_VALUES[@]}")
fi

OPT_ADAMW8BIT=(
    --optimizer_type adamw8bit
    --learning_rate 2e-4
)

OPT_ADAFACTOR=(
    --optimizer_type adafactor
    --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False"
    --lr_scheduler constant
    --learning_rate 2e-4
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
    prodigy)   OPT_ARGS=("${OPT_PRODIGY[@]}") ;;
    *) echo "Unknown OPTIMIZER: $OPTIMIZER" >&2; exit 1 ;;
esac

DATASET_TOML=".pyro/krea2/cfg/${NAME}.toml"
PROMPT_TOML=".pyro/krea2/cfg/p_${NAME}.toml"

for path in "$DIT" "$TENC" "$VAE" "$TURBO_LORA" "$DATASET_TOML" "$PROMPT_TOML"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

if [[ -n "$BYPASS" && ! -f "$BYPASS" ]]; then
    echo "Missing Krea2 bypass file: $BYPASS" >&2
    exit 1
fi

BLOCK_SWAP_ARGS=()
if (( BLOCKS_TO_SWAP > 0 )); then
    BLOCK_SWAP_ARGS=(--blocks_to_swap "$BLOCKS_TO_SWAP" --use_pinned_memory_for_block_swap)
fi

SAMPLE_BLOCK_SWAP_ARGS=()
if [[ "$SAMPLE_BLOCKS_TO_SWAP" != "inherit" ]]; then
    SAMPLE_BLOCK_SWAP_ARGS=(--sample_blocks_to_swap "$SAMPLE_BLOCKS_TO_SWAP")
fi

SAMPLE_OFFLOAD_ARGS=()
if (( SAMPLE_WITH_OFFLOADING_ENABLED )); then
    SAMPLE_OFFLOAD_ARGS=(--sample_with_offloading)
fi

BYPASS_ARGS=()
if [[ -n "$BYPASS" ]]; then
    BYPASS_ARGS=(--bypass "$BYPASS" --bypass-weight "$BYPASS_WEIGHT")
fi

source .venv/bin/activate

echo "Krea2 training run: dataset=$NAME output=$RUN_NAME lora_config=$LORA_CONFIG"
echo "Krea2 cache step: CACHE_DATASET=$CACHE_DATASET"
echo "Krea2 sample decode offload: SAMPLE_WITH_OFFLOADING=$SAMPLE_WITH_OFFLOADING"
if [[ -n "$BYPASS" ]]; then
    echo "Krea2 projector bypass: BYPASS=$BYPASS BYPASS_WEIGHT=$BYPASS_WEIGHT"
fi
if ((${#NETWORK_ARGS_VALUES[@]} > 0)); then
    printf 'Krea2 LoRA network args:'
    printf ' %q' "${NETWORK_ARGS_VALUES[@]}"
    printf '\n'
fi

if (( CACHE_DATASET_ENABLED )); then
    python src/musubi_tuner/krea2_cache_latents.py \
        --dataset_config "$DATASET_TOML" \
        --vae "$VAE" \
        --batch_size "$CACHE_LATENTS_BATCH_SIZE"

    python src/musubi_tuner/krea2_cache_text_encoder_outputs.py \
        --dataset_config "$DATASET_TOML" \
        --text_encoder "$TENC" \
        --batch_size "$CACHE_TEXT_BATCH_SIZE"
fi

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/krea2_train_network.py \
    --dit "$DIT" \
    --vae "$VAE" \
    --text_encoder "$TENC" \
    --dataset_config "$DATASET_TOML" \
    --sample_prompts "$PROMPT_TOML" \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling krea2_shift --weighting_scheme none \
    "${OPT_ARGS[@]}" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_krea2 --network_dim "$NETWORK_DIM" --network_alpha "$NETWORK_ALPHA" \
    "${NETWORK_ARGS_CLI[@]}" \
    "${BYPASS_ARGS[@]}" \
    --seed 42 \
    --save_every_n_steps "$SAVE_EVERY" --max_train_steps "$MAX_STEPS" \
    --save_state --save_last_n_steps_state 2 --autoresume \
    --output_dir /home/pyro/models/_out/krea2/"$RUN_NAME" \
    --output_name "$RUN_NAME" \
    --logging_dir /home/pyro/models/_out/krea2/.logs \
    --sampling_lora_weight "$TURBO_LORA" \
    --sampling_lora_multiplier 1.0 \
    --fp8_base --fp8_scaled \
    "${BLOCK_SWAP_ARGS[@]}" \
    "${SAMPLE_BLOCK_SWAP_ARGS[@]}" \
    "${SAMPLE_OFFLOAD_ARGS[@]}" \
    --sample_at_first --sample_every_n_steps "$SAMPLE_EVERY" --log_with trackio
