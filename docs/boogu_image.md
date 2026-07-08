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
- Edit-style preview sampling with pre-cached sample input images.
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

### Sample Prompt Input Images

Boogu sample prompt TOML can attach one optional edit input image for preview sampling. Use `input_image` under `[prompt]` to apply it to every subset, or under an individual `[[prompt.subset]]` to override it for that prompt. Relative paths resolve from the prompt TOML directory.

```toml
[prompt]
width = 1024
height = 1024
sample_steps = 4
cfg_scale = 1.0
guidance_scale = 1.0
boogu_sampler = "dmd"
input_image = "source.png"

[[prompt.subset]]
prompt = "make the person do a straight chest stand"

[[prompt.subset]]
prompt = "make the person do a side plank"
input_image = "alternate-source.png"
```

For Flux/Klein-style prompt files, a string or single-item `control_image_path` is accepted as a compatibility alias when `input_image` is not present. Multiple input images are rejected; Boogu preview sampling currently supports one source image per prompt.

Input images are pre-cached before transformer loading: Qwen3-VL receives the resized image for positive instruction features, and the VAE reference latent is reused during snapshot denoising. Dataset training batches remain output-only.

Use `boogu_sampler = "dmd"` for turbo edit preview sampling. It follows Boogu's few-step DMD path and requires the no-CFG shape `cfg_scale = 1.0` with `guidance_scale = 1.0`. Omit `boogu_sampler` for the default flow sampler used by the base/edit checkpoints.

### Edit Prompting Guide

Boogu edit preview prompts are instruction prompts, not plain final-image captions. With `input_image`, Qwen3-VL receives the source image and the prompt under an edit system prompt, so wording like `make her ...`, `transform this image ...`, or `use the input as reference ...` usually gives stronger edits than a caption such as `a woman sitting on a chair`.

Use a preservation budget when you want broader edits without specifying every detail:

```text
make [subject] [do/change X]. Preserve [A, B]. Freely change everything else.
```

Common patterns:

```text
make Yuna sit naturally on a simple chair. Preserve her face, hairstyle, and neon-city identity. Freely change pose, hands, composition, camera angle, and clothing folds as needed. No weapon, no handheld prop.
```

```text
use the input image only as a reference for Yuna's identity and style. Create a new image where she is sitting casually on a simple chair.
```

```text
transform the image so Yuna is sitting on a simple chair. Keep the same character and visual style, but allow major changes to pose, background, and framing.
```

Edit-strength tiers:

```text
Surgical: make her dress blue. Preserve pose, face, hands, background, lighting, and composition.

Medium: make her sit on a chair. Preserve face, hairstyle, outfit style, and neon-city mood. Adjust pose and composition as needed.

Free: make Yuna sitting casually on a chair. Use the input only as character/style reference. Freely change pose, camera angle, background, and outfit details.

Wild: reinterpret Yuna as a relaxed character portrait of her sitting on a chair in a new scene. Preserve only her identity and anime style.
```

For the turbo/DMD path, avoid caption-only prompts when you expect an edit. Prefer imperative/reference language plus explicit freedom and preservation clauses: `Preserve only X; freely change Y`.

### Standalone Prompt File Inference

Use `boogu_image_generate_image.py` to run a sample prompt file without starting a training job. This uses the same prompt/image cache and preview sampler as training.

```bash
python src/musubi_tuner/boogu_image_generate_image.py \
  --dit /home/pyro/models/comfy/diffusion_models/boogu_image_edit_turbo_bf16.safetensors \
  --vae /home/pyro/models/comfy/vae/ae.safetensors \
  --text_encoder /home/pyro/models/comfy/text_encoders/qwen3vl_8b_fp8_scaled.safetensors \
  --processor /home/pyro/models/qwen3-vl-4b \
  --sample_prompts .pyro/boogu/cfg/p_test.toml \
  --output_dir /home/pyro/models/_out/boogu/yuna_test \
  --output_name yuna_test \
  --mixed_precision bf16 --sdpa --fp8_base --fp8_scaled
```

Outputs are written under `<output_dir>/sample/`.

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
