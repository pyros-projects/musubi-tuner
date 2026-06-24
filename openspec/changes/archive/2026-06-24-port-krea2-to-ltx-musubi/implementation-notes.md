## Baseline

- Branch before Krea2 code edits: `ltx-2...pyros-projects/ltx-2`.
- Pre-existing dirty files before Krea2 code edits:
  - `src/musubi_tuner/dataset/image_video_dataset.py`
  - `src/musubi_tuner/hv_train_network.py`
  - `src/musubi_tuner/ltx2_train_network.py`
  - `src/musubi_tuner/qwen_image_train_network.py`
  - `tests/test_compile_prewarm.py`
  - `tests/test_ltx2_sampling_lora.py`
  - `tests/test_remaining_migration_ports.py`
  - `src/musubi_tuner/prompt_lora_utils.py`
- OpenSpec initialization for this change added `.codex/` and `openspec/`.
- Upstream Krea2 source is available at `/tmp/krea2-fp8-spelunk/musubi-tuner` on `main...origin/main`.

## Import Layout Notes

- Upstream Krea2 uses newer split modules: `dataset/architectures.py`, `dataset/cache_io.py`, `training/parser_common.py`, and `training/trainer_base.py`.
- LTX keeps architecture constants, cache writers, parser helpers, and `NetworkTrainer` in `dataset/image_video_dataset.py` and `hv_train_network.py`.
- Upstream Krea2 trainer returns `DiTOutput`; LTX model trainers return `(pred, target)` tuples.
- Top-level wrapper convention matches Qwen: import `main` from `musubi_tuner.<script>` and call it.

## Local Model Paths

- RAW/base DiT: `/home/pyro/models/comfy/diffusion_models/krea2_raw_bf16.safetensors`
- Qwen-Image VAE: `/home/pyro/models/comfy/vae/qwen_image_vae.safetensors`
- Qwen3-VL text encoder: `/home/pyro/models/comfy/text_encoders/qwen3vl_4b_bf16.safetensors`
- Turbo LoRA sampling adapter: `/home/pyro/models/comfy/loras/krea/krea2_turbo_lora_rank_64_bf16.safetensors`

## Verification

- `.venv/bin/python -m compileall -q src/musubi_tuner/krea2 src/musubi_tuner/krea2_train_network.py src/musubi_tuner/krea2_generate_image.py src/musubi_tuner/krea2_cache_latents.py src/musubi_tuner/krea2_cache_text_encoder_outputs.py src/musubi_tuner/networks/lora_krea2.py` passed.
- Import check passed for `Qwen2TokenizerFast`, `Qwen3VLConfig`, `Qwen3VLForConditionalGeneration`, `musubi_tuner.krea2_train_network`, `musubi_tuner.krea2_generate_image`, `musubi_tuner.qwen_image_train_network`, `musubi_tuner.zimage_train_network`, and `musubi_tuner.ltx2_train_network`.
- `.venv/bin/python -m pytest tests/test_krea2_gather_valid_text.py tests/test_krea2_timesteps.py tests/test_krea2_sampling_lora.py -q` passed: 10 tests.
- CLI help/import checks passed for `krea2_train_network.py`, `krea2_cache_latents.py`, `krea2_cache_text_encoder_outputs.py`, and `krea2_generate_image.py`.
- GPU/model-weight smoke tests were intentionally not run because the GPU was in use.

## Manual Smoke Commands

Use a real image dataset config in place of `/path/to/krea2_dataset.toml`.

Cache latents:

```bash
.venv/bin/python src/musubi_tuner/krea2_cache_latents.py \
  --dataset_config /path/to/krea2_dataset.toml \
  --vae /home/pyro/models/comfy/vae/qwen_image_vae.safetensors \
  --vae_dtype bfloat16
```

Cache text embeddings:

```bash
.venv/bin/python src/musubi_tuner/krea2_cache_text_encoder_outputs.py \
  --dataset_config /path/to/krea2_dataset.toml \
  --text_encoder /home/pyro/models/comfy/text_encoders/qwen3vl_4b_bf16.safetensors \
  --text_encoder_dtype bfloat16 \
  --batch_size 1
```

Minimal one-step RAW/base LoRA train with Turbo-LoRA snapshot preview:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/krea2_train_network.py \
  --dit /home/pyro/models/comfy/diffusion_models/krea2_raw_bf16.safetensors \
  --vae /home/pyro/models/comfy/vae/qwen_image_vae.safetensors \
  --text_encoder /home/pyro/models/comfy/text_encoders/qwen3vl_4b_bf16.safetensors \
  --dataset_config /path/to/krea2_dataset.toml \
  --sdpa --mixed_precision bf16 --fp8_base --fp8_scaled \
  --timestep_sampling krea2_shift --weighting_scheme none \
  --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
  --max_train_steps 1 --save_every_n_steps 1 --seed 42 \
  --network_module musubi_tuner.networks.lora_krea2 --network_dim 32 --network_alpha 32 \
  --sampling_lora_weight /home/pyro/models/comfy/loras/krea/krea2_turbo_lora_rank_64_bf16.safetensors \
  --sampling_lora_multiplier 1.0 \
  --sample_prompts /path/to/krea2_sample_prompts.txt --sample_every_n_steps 1 \
  --output_dir /home/pyro/models/_out/krea-musubi-smoke --output_name krea2_smoke
```

Sample prompt line for Turbo preview snapshots:

```text
A fox in the snow --w 1024 --h 1024 --s 8 --l 1 --mu 1.15 --d 0
```
