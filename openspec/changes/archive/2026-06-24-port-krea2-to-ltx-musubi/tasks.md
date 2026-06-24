## 1. Baseline And Port Inputs

- [x] 1.1 Capture current `ltx-musubi` git status and note pre-existing dirty files before editing.
- [x] 1.2 Reconfirm upstream Krea2 source paths in `/tmp/krea2-fp8-spelunk/musubi-tuner` or refresh the clean upstream clone if missing.
- [x] 1.3 Inspect upstream Krea2 files and LTX Qwen/Image helper files for import-layout differences before copying.

## 2. Add Isolated Krea2 Files

- [x] 2.1 Copy/adapt `src/musubi_tuner/krea2/` from upstream into LTX.
- [x] 2.2 Copy/adapt `src/musubi_tuner/networks/lora_krea2.py` into LTX.
- [x] 2.3 Copy/adapt package-level wrappers `krea2_cache_latents.py`, `krea2_cache_text_encoder_outputs.py`, `krea2_train_network.py`, and `krea2_generate_image.py`.
- [x] 2.4 Copy/adapt root-level wrapper scripts if LTX convention requires top-level launch files.

## 3. Dataset And Cache Glue

- [x] 3.1 Add Krea2 architecture constants to `src/musubi_tuner/dataset/image_video_dataset.py`.
- [x] 3.2 Add Krea2 16-pixel resolution step mapping to the existing `BucketSelector`.
- [x] 3.3 Add `save_latent_cache_krea2()` to the LTX dataset cache helpers.
- [x] 3.4 Add `save_text_encoder_output_cache_krea2()` to the LTX dataset cache helpers.
- [x] 3.5 Adapt Krea2 cache scripts to import architecture/cache helpers from `image_video_dataset.py` rather than upstream split modules.
- [x] 3.6 Verify Krea2 cache metadata uses architecture `krea2` and compatible dtype-suffixed keys.

## 4. Trainer And Sampling Integration

- [x] 4.1 Adapt upstream `Krea2NetworkTrainer` to inherit from LTX `NetworkTrainer` in `hv_train_network.py`.
- [x] 4.2 Implement Krea2 model-specific argument handling, including bf16 defaults and plain-fp8 rejection.
- [x] 4.3 Implement Krea2 DiT/VAE loading using Qwen-Image VAE helpers and Krea2 scaled-fp8 loader.
- [x] 4.4 Implement Krea2 training `call_dit`/forward path for cached latents and `krea2_vl_embed` varlen text embeddings.
- [x] 4.5 Implement Krea2 sample prompt processing so Qwen3-VL embeds are cached and the text encoder is released before training.
- [x] 4.6 Implement Krea2 snapshot denoising and VAE decode using cached prompt embeddings.
- [x] 4.7 Add Krea2 `mu` sample override support for fixed `mu=1.15` Turbo-preview snapshots.

## 5. Sampling LoRA Adapter Support

- [x] 5.1 Set Krea2's default sampling LoRA network module to `musubi_tuner.networks.lora_krea2`.
- [x] 5.2 Add Krea2 `normalize_sampling_lora_weights()` support for native `diffusion_model.*` Turbo LoRA keys.
- [x] 5.3 Ensure normalized local Turbo LoRA keys map to `lora_unet_*` keys expected by `networks.lora_krea2`.
- [x] 5.4 Verify Krea2 sampling LoRAs use the existing global and prompt-local application/restore paths.
- [x] 5.5 Confirm sampling-LoRA application works with fp8 modules through the existing runtime overlay fallback.

## 6. Shared Trainer And Attention Glue

- [x] 6.1 Add `krea2_shift` to LTX `--timestep_sampling` parser choices.
- [x] 6.2 Add `krea2_shift` timestep computation using Krea2 sequence-length endpoints for 256px and 1280px images.
- [x] 6.3 Port upstream shared attention GQA handling for Krea2 query/key-value head mismatch.
- [x] 6.4 Port upstream `reshape()` fixes for non-contiguous flash/sage varlen attention tensors.
- [x] 6.5 Add Krea2 SAI/model metadata support if needed by checkpoint save/conversion.

## 7. Dependency And Import Compatibility

- [x] 7.1 Update dependency metadata so Qwen3-VL classes are available, most likely by bumping `transformers` to `4.57.6`.
- [x] 7.2 Verify `Qwen3VLConfig`, `Qwen3VLForConditionalGeneration`, and `Qwen2TokenizerFast` import in the LTX environment.
- [x] 7.3 Avoid wholesale upstream fp8 utility replacement; port only the utility changes required by Krea2.
- [x] 7.4 Run import checks for Krea2, Qwen Image, Z-Image, and LTX2 modules after dependency changes.

## 8. Documentation And Examples

- [x] 8.1 Add/update `docs/krea2.md` for the LTX port.
- [x] 8.2 Document RAW/base training plus Turbo-LoRA sampling adapter preview as the supported workflow.
- [x] 8.3 Document `--fp8_base --fp8_scaled`, Krea2 LoRA defaults, `krea2_shift`, and snapshot `mu=1.15`.
- [x] 8.4 Mark upstream full Turbo-DiT swapping as out of initial scope or unsupported in this fork.

## 9. Tests

- [x] 9.1 Add/adapt upstream `test_krea2_gather_valid_text.py`.
- [x] 9.2 Add/adapt upstream `test_krea2_timesteps.py` for LTX's monolithic trainer layout.
- [x] 9.3 Add a CPU test for Krea2 sampling-LoRA native `diffusion_model.*` key normalization.
- [x] 9.4 Add lightweight import/parser tests for Krea2 scripts where feasible without model weights.
- [x] 9.5 Run the relevant CPU test subset and record any skipped GPU/model-weight checks.

## 10. Manual Smoke Instructions

- [x] 10.1 Provide commands to cache Krea2 latents and text embeddings with local model paths.
- [x] 10.2 Provide a minimal one-step Krea2 LoRA training command using RAW/base DiT.
- [x] 10.3 Provide a minimal snapshot command/config using the local Turbo LoRA as `--sampling_lora_weight`.
- [x] 10.4 Note that GPU smoke tests should be run by Pyro once the GPU is free.
