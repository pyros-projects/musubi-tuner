# Boogu Image

This document covers the experimental Boogu Image Base text-to-image LoRA path in Musubi Tuner.

Boogu Image is a Lumina2-style mixed-stream flow-matching image model conditioned on Qwen3-VL instruction features. The current Musubi integration targets **Boogu/Boogu-Image-0.1-Base** text-to-image training only.

Reference implementation: `ostris/ai-toolkit` commit `4a99ddabadbb27e5471d7023c9b429b5e0b39cb6`, under `extensions_built_in/diffusion_models/boogu_image/`.

## Scope

Supported now:

- Base text-to-image LoRA training.
- Cached FLUX-compatible AutoencoderKL latents.
- Cached natural-length Qwen3-VL instruction features.
- In-training preview sampling from cached prompt embeddings.
- ComfyUI companion LoRA export as `*.comfy.safetensors`.
- Scaled fp8 transformer loading from a bf16 `.safetensors` checkpoint.
- Turbo LoRA preview attachment for the local smoke helper.

Not supported yet:

- Boogu Edit / TI2I training.
- Direct torchao `-fp8` `.bin` training checkpoints.
- Boogu block swap.
- Full Turbo LoRA quality/parity validation. The local smoke helper attaches the Turbo LoRA for a fast preview path, but the GPU smoke is still deferred.

## Local Paths

Pyro's current local candidates:

```text
DiT  /home/pyro/models/comfy/diffusion_models/boogu_image_base_bf16.safetensors
VAE  /home/pyro/models/comfy/vae/ae.safetensors
VAE  /home/pyro/models/comfy/vae/flux1_vae_bf16.safetensors
TE   /home/pyro/models/comfy/text_encoders/qwen3vl_8b_fp8_scaled.safetensors
PROC /home/pyro/models/qwen3-vl-4b
LoRA /home/pyro/models/comfy/loras/boogu/boogu_image_turbo_lora_rank_128_bf16.safetensors
```

For another machine, set these paths explicitly in commands or override the environment variables used by `.pyro/boogu/train.sh`.

## Cache Latents

```bash
python src/musubi_tuner/boogu_image_cache_latents.py \
  --dataset_config path/to/dataset.toml \
  --vae /home/pyro/models/comfy/vae/ae.safetensors \
  --batch_size 1
```

The cache stores one tensor per item with a key like `latents_64x64_bfloat16` and metadata `architecture=boogu_image`.

## Cache Text Encoder Outputs

```bash
python src/musubi_tuner/boogu_image_cache_text_encoder_outputs.py \
  --dataset_config path/to/dataset.toml \
  --text_encoder /home/pyro/models/comfy/text_encoders/qwen3vl_8b_fp8_scaled.safetensors \
  --processor /home/pyro/models/qwen3-vl-4b \
  --batch_size 1
```

The cache stores natural-length Qwen3-VL-8B instruction features with a key like `varlen_boogu_instruction_embed_bfloat16`. Training consumes these cached tensors and does not load Qwen3-VL in the training loop. For single-file ComfyUI text encoder weights, `--processor` supplies tokenizer/processor assets only.

## Train

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/boogu_image_train_network.py \
  --dit /home/pyro/models/comfy/diffusion_models/boogu_image_base_bf16.safetensors \
  --vae /home/pyro/models/comfy/vae/ae.safetensors \
  --dataset_config path/to/dataset.toml \
  --sample_prompts path/to/prompts.toml \
  --text_encoder /home/pyro/models/comfy/text_encoders/qwen3vl_8b_fp8_scaled.safetensors \
  --processor /home/pyro/models/qwen3-vl-4b \
  --sdpa --mixed_precision bf16 \
  --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 3.0 \
  --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
  --network_module networks.lora_boogu_image --network_dim 16 --network_alpha 16 \
  --fp8_base --fp8_scaled \
  --max_train_steps 1 --save_every_n_steps 1 \
  --output_dir /home/pyro/models/_out/boogu/smoke \
  --output_name boogu-smoke \
  --sample_at_first --sample_every_n_steps 1
```

`--fp8_base --fp8_scaled` must be used together. Plain fp8 is rejected because norms and output-sensitive modules must stay in a safe dtype.

## LoRA Export

By default, saved Boogu LoRA checkpoints emit:

- `name.safetensors`: Musubi training/resume format.
- `name.comfy.safetensors`: ComfyUI-compatible `diffusion_model.*` LoRA keys.

Use `--no_convert_to_comfy` to skip the companion export, or `--no_save_original_lora` to keep only the ComfyUI file. Keep the original Musubi file if you may resume training.

Manual conversion:

```bash
python src/musubi_tuner/boogu_image/convert_lora_to_comfy.py /path/to/boogu_lora.safetensors
```

## Deferred GPU Smoke

The GPU was busy during implementation, so these are the deferred checks to run when it is free:

```bash
CACHE_DATASET=1 SMOKE=1 .pyro/boogu/train.sh
```

Expanded smoke sequence:

```bash
python src/musubi_tuner/boogu_image_cache_latents.py \
  --dataset_config .pyro/boogu/cfg/smoke.toml \
  --vae /home/pyro/models/comfy/vae/ae.safetensors \
  --batch_size 1

python src/musubi_tuner/boogu_image_cache_text_encoder_outputs.py \
  --dataset_config .pyro/boogu/cfg/smoke.toml \
  --text_encoder /home/pyro/models/comfy/text_encoders/qwen3vl_8b_fp8_scaled.safetensors \
  --processor /home/pyro/models/qwen3-vl-4b \
  --batch_size 1

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/boogu_image_train_network.py \
  --dit /home/pyro/models/comfy/diffusion_models/boogu_image_base_bf16.safetensors \
  --vae /home/pyro/models/comfy/vae/ae.safetensors \
  --text_encoder /home/pyro/models/comfy/text_encoders/qwen3vl_8b_fp8_scaled.safetensors \
  --processor /home/pyro/models/qwen3-vl-4b \
  --dataset_config .pyro/boogu/cfg/smoke.toml \
  --sample_prompts .pyro/boogu/cfg/p_smoke.toml \
  --sdpa --mixed_precision bf16 \
  --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 3.0 \
  --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
  --network_module networks.lora_boogu_image --network_dim 16 --network_alpha 16 \
  --sampling_lora_weight /home/pyro/models/comfy/loras/boogu/boogu_image_turbo_lora_rank_128_bf16.safetensors \
  --sampling_lora_multiplier 1.0 \
  --fp8_base --fp8_scaled \
  --max_train_steps 1 --save_every_n_steps 1 \
  --output_dir /home/pyro/models/_out/boogu/smoke \
  --output_name boogu-smoke \
  --sample_at_first --sample_every_n_steps 1
```
