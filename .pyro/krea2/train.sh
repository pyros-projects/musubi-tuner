#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# User-tunable values. Every value can be overridden from the environment:
#   NAME=emo SAMPLE_EVERY=25 .pyro/krea2/train.sh
DIT="${DIT:-/home/pyro/models/comfy/diffusion_models/krea2_raw_bf16.safetensors}"
TENC="${TENC:-/home/pyro/models/comfy/text_encoders/qwen3vl_4b_bf16.safetensors}"
VAE="${VAE:-/home/pyro/models/comfy/vae/qwen_image_vae.safetensors}"
TURBO_LORA="${TURBO_LORA:-/home/pyro/models/comfy/loras/krea/krea2_turbo_lora_rank_64_bf16.safetensors}"

NAME="${NAME:-cobra}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"   # adamw8bit | adafactor | prodigy
SAMPLE_EVERY="${SAMPLE_EVERY:-50}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-100}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NETWORK_DIM="${NETWORK_DIM:-16}"
NETWORK_ALPHA="${NETWORK_ALPHA:-16}"
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-10}"
SAMPLE_BLOCKS_TO_SWAP="${SAMPLE_BLOCKS_TO_SWAP:-0}"  # 0 = unswapped snapshots, "inherit" = use BLOCKS_TO_SWAP

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

BLOCK_SWAP_ARGS=()
if (( BLOCKS_TO_SWAP > 0 )); then
    BLOCK_SWAP_ARGS=(--blocks_to_swap "$BLOCKS_TO_SWAP" --use_pinned_memory_for_block_swap)
fi

SAMPLE_BLOCK_SWAP_ARGS=()
if [[ "$SAMPLE_BLOCKS_TO_SWAP" != "inherit" ]]; then
    SAMPLE_BLOCK_SWAP_ARGS=(--sample_blocks_to_swap "$SAMPLE_BLOCKS_TO_SWAP")
fi

source .venv/bin/activate

python src/musubi_tuner/krea2_cache_latents.py \
    --dataset_config "$DATASET_TOML" \
    --vae "$VAE" \
    --batch_size 2

python src/musubi_tuner/krea2_cache_text_encoder_outputs.py \
    --dataset_config "$DATASET_TOML" \
    --text_encoder "$TENC" \
    --batch_size 1

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
    --seed 42 \
    --save_every_n_steps "$SAVE_EVERY" --max_train_steps "$MAX_STEPS" \
    --save_state --save_last_n_steps_state 2 --autoresume \
    --output_dir /home/pyro/models/_out/krea2/"$NAME" \
    --output_name "$NAME" \
    --logging_dir /home/pyro/models/_out/krea2/.logs \
    --sampling_lora_weight "$TURBO_LORA" \
    --sampling_lora_multiplier 1.0 \
    --fp8_base --fp8_scaled \
    "${BLOCK_SWAP_ARGS[@]}" \
    "${SAMPLE_BLOCK_SWAP_ARGS[@]}" \
    --sample_at_first --sample_every_n_steps "$SAMPLE_EVERY" --log_with trackio
