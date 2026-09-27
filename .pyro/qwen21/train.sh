#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# User-tunable values. Every value can be overridden from the environment:
#   NAME=emo SAMPLE_EVERY=25 .pyro/qwen21/train.sh
#   NAME=msplits LORA_CONFIG=preset-1 .pyro/qwen21/train.sh
#   NAME=msplits CACHE_DATASET=1 .pyro/qwen21/train.sh
#   NAME=msplits SAMPLE_WITH_OFFLOADING=0 .pyro/qwen21/train.sh
#   NAME=msplits TURBO_LORA= .pyro/qwen21/train.sh   (snapshots on the base model with the prompt file's sample_steps)
DIT="${DIT:-/home/pyro/models/comfy/diffusion_models/qwen_image_2.1_int8_convrot.safetensors}"
TENC="${TENC:-/home/pyro/models/comfy/text_encoders/qwen3vl_8b_int8_convrot.safetensors}"
VAE="${VAE:-/home/pyro/models/comfy/vae/qwen_image_2.1_vae_bf16.safetensors}"
# Snapshot-only turbo LoRA and its raw sigma nodes (Viggle v0.2.1: 6 steps). Set TURBO_LORA= to sample without it.
TURBO_LORA="${TURBO_LORA-/home/pyro/models/comfy/loras/qwen21/turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors}"
TURBO_SIGMAS="${TURBO_SIGMAS:-1.0,0.9375,0.875,0.75,0.5,0.25}"

NAME="${NAME:-flat}"
LORA_CONFIG="${LORA_CONFIG:-default}"   # default | preset-1 | preset-2 | preset-3 | custom
NETWORK_ARGS="${NETWORK_ARGS:-}"        # optional raw Musubi --network_args entries, e.g. "verbose=True"
CACHE_DATASET="${CACHE_DATASET:-1}"     # 1/true/yes/on = cache latents and text encoder outputs before training
OPTIMIZER="${OPTIMIZER:-adamw8bit}"   # adamw8bit | adafactor | prodigy
SAMPLE_EVERY="${SAMPLE_EVERY:-50}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-100}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NETWORK_DIM="${NETWORK_DIM:-16}"
NETWORK_ALPHA="${NETWORK_ALPHA:-16}"
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
        # Qwen-Image 2.1 module default: every Linear in the DiT (blocks, embedders, shared modulation, output).
        ;;
    preset-1)
        # Main per-block attention only: to_q/to_k/to_v/to_out.0.
        LORA_PRESET_ARGS=("include_patterns=['transformer_blocks\\.[0-9]+\\.attn\\.(to_q|to_k|to_v|to_out\\.0)']")
        ;;
    preset-2)
        # Main per-block attention + fused SwiGLU MLP (gate_up/out), no embedders/modulation/output.
        LORA_PRESET_ARGS=("include_patterns=['transformer_blocks\\.[0-9]+\\.(attn\\.(to_q|to_k|to_v|to_out\\.0)|img_mlp\\.(gate_up|out))']")
        ;;
    preset-3)
        # Diffusion-pipe-ish: any Linear module whose path contains "blocks." (same set as preset-2 here).
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

DATASET_TOML=".pyro/qwen21/cfg/${NAME}.toml"
PROMPT_TOML=".pyro/qwen21/cfg/p_${NAME}.toml"

for path in "$DIT" "$TENC" "$VAE" "$DATASET_TOML" "$PROMPT_TOML"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

TURBO_ARGS=()
if [[ -n "$TURBO_LORA" ]]; then
    if [[ ! -f "$TURBO_LORA" ]]; then
        echo "Missing turbo LoRA: $TURBO_LORA (set TURBO_LORA= to sample without one)" >&2
        exit 1
    fi
    TURBO_ARGS=(--sampling_lora_weight "$TURBO_LORA" --sampling_lora_multiplier 1.0 --sample_raw_sigmas "$TURBO_SIGMAS")
fi

SAMPLE_OFFLOAD_ARGS=()
if (( SAMPLE_WITH_OFFLOADING_ENABLED )); then
    SAMPLE_OFFLOAD_ARGS=(--sample_with_offloading)
fi

source .venv/bin/activate

echo "Qwen-Image 2.1 training run: dataset=$NAME output=$RUN_NAME lora_config=$LORA_CONFIG"
echo "Qwen-Image 2.1 cache step: CACHE_DATASET=$CACHE_DATASET"
echo "Qwen-Image 2.1 sample decode offload: SAMPLE_WITH_OFFLOADING=$SAMPLE_WITH_OFFLOADING"
if [[ -n "$TURBO_LORA" ]]; then
    echo "Qwen-Image 2.1 turbo snapshots: TURBO_LORA=$TURBO_LORA TURBO_SIGMAS=$TURBO_SIGMAS"
else
    echo "Qwen-Image 2.1 turbo snapshots: off (base model, prompt file sample_steps)"
fi
if ((${#NETWORK_ARGS_VALUES[@]} > 0)); then
    printf 'Qwen-Image 2.1 LoRA network args:'
    printf ' %q' "${NETWORK_ARGS_VALUES[@]}"
    printf '\n'
fi

if (( CACHE_DATASET_ENABLED )); then
    python src/musubi_tuner/qwen_image21_cache_latents.py \
        --dataset_config "$DATASET_TOML" \
        --vae "$VAE" \
        --batch_size "$CACHE_LATENTS_BATCH_SIZE"

    python src/musubi_tuner/qwen_image21_cache_text_encoder_outputs.py \
        --dataset_config "$DATASET_TOML" \
        --text_encoder "$TENC" \
        --batch_size "$CACHE_TEXT_BATCH_SIZE"
fi

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/qwen_image21_train_network.py \
    --dit "$DIT" \
    --vae "$VAE" \
    --text_encoder "$TENC" \
    --dataset_config "$DATASET_TOML" \
    --sample_prompts "$PROMPT_TOML" \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling qwen21_shift --weighting_scheme none \
    "${OPT_ARGS[@]}" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_qwen_image21 --network_dim "$NETWORK_DIM" --network_alpha "$NETWORK_ALPHA" \
    "${NETWORK_ARGS_CLI[@]}" \
    --seed 42 \
    --save_every_n_steps "$SAVE_EVERY" --max_train_steps "$MAX_STEPS" \
    --save_state --save_last_n_steps_state 2 --autoresume \
    --output_dir /home/pyro/models/_out/qwen21/"$RUN_NAME" \
    --output_name "$RUN_NAME" \
    --logging_dir /home/pyro/models/_out/qwen21/.logs \
    "${TURBO_ARGS[@]}" \
    "${SAMPLE_OFFLOAD_ARGS[@]}" \
    --sample_at_first --sample_every_n_steps "$SAMPLE_EVERY" --log_with trackio
