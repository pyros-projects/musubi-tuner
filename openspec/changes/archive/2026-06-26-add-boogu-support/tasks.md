## 1. Baseline And References

- [x] 1.1 Capture current `ltx-musubi` git status and note pre-existing dirty files before editing. Current status before reference edits: clean `ltx-2` branch tracking `pyros-projects/ltx-2`.
- [x] 1.2 Record the ai-toolkit Boogu reference commit and inspect the remote files under `extensions_built_in/diffusion_models/boogu_image/`. Use `https://github.com/ostris/ai-toolkit` commit `4a99ddabadbb27e5471d7023c9b429b5e0b39cb6` as the pinned reference; Boogu landed in `60c1ac6`, and the Boogu directory had no diffs between those commits when inspected on 2026-06-26.
- [x] 1.3 Inspect the official Boogu repository/model card for VAE, scheduler, text encoder, fp8, and supported-model-scope details. Official sources inspected on 2026-06-26: `boogu-project/Boogu-Image` commit `13853b77b6ba8dd393685231b662acf9f90730e9` and HF `Boogu/Boogu-Image-0.1-Base` SHA `e10ed9d3691f25416d439aeb4de33b22bf9039c1`; details recorded in design.md.
- [x] 1.4 Confirm local model availability or document expected paths for Boogu Base transformer, VAE, and Qwen3-VL components. Local paths used by the successful `lucy-adamw8bit` run: `/home/pyro/models/comfy/diffusion_models/boogu_image_base_bf16.safetensors`, `/home/pyro/models/comfy/vae/ae.safetensors`, `/home/pyro/models/comfy/text_encoders/qwen3vl_8b_fp8_scaled.safetensors`, `/home/pyro/models/qwen3-vl-4b` for processor/tokenizer assets, and `/home/pyro/models/comfy/loras/boogu/boogu_image_turbo_lora_rank_128_bf16.safetensors` for turbo snapshot sampling.

## 2. TDD Harness First

- [x] 2.1 Add failing CPU tests for Boogu parser/import registration and architecture constants.
- [x] 2.2 Add failing CPU tests for Boogu latent/text cache metadata and dtype-key conventions.
- [x] 2.3 Add failing CPU tests for variable-length instruction feature padding and attention masks.
- [x] 2.4 Add failing CPU tests for Boogu time-shift schedule and Musubi-to-Boogu velocity sign conversion.
- [x] 2.5 Add failing CPU tests for Boogu LoRA target discovery on a tiny fake Boogu-like module tree.
- [x] 2.6 Add failing CPU tests for Boogu Musubi-to-native/Comfy LoRA key conversion and alpha handling.
- [x] 2.7 Add failing CPU tests for Boogu post-save hook behavior with `convert_to_comfy` and `save_original_lora` flags.

## 3. Isolated Boogu Model Files

- [x] 3.1 Add `src/musubi_tuner/boogu_image/` package with transformer, attention processor, Lumina block, embedding, RoPE, and pipeline helpers adapted from ai-toolkit/Boogu.
- [x] 3.2 Remove or omit inference-only Boogu code paths that are not needed for training, including TeaCache, TaylorSeer, and prompt rewriting.
- [x] 3.3 Adapt imports, dtype handling, device movement, and attention backend use to the LTX Musubi package layout.
- [x] 3.4 Add lightweight import tests for the new Boogu package modules.

## 4. Dataset And Cache Glue

- [x] 4.1 Add Boogu architecture constants and full-name metadata to `src/musubi_tuner/dataset/image_video_dataset.py`.
- [x] 4.2 Add Boogu bucket divisibility/resolution-step handling compatible with VAE scale 8 and patch size 2.
- [x] 4.3 Add `save_latent_cache_boogu_image()` for Boogu latent tensors and metadata.
- [x] 4.4 Add `save_text_encoder_output_cache_boogu_image()` for variable-length instruction features.
- [x] 4.5 Implement `boogu_image_cache_latents.py` using the Boogu-compatible AutoencoderKL VAE and normalized scaling/shift convention.
- [x] 4.6 Implement `boogu_image_cache_text_encoder_outputs.py` using Qwen3-VL instruction features and natural-length tensor storage.
- [x] 4.7 Run the cache metadata/padding tests and update them only if the real cache contract intentionally differs.

## 5. LoRA Module And Conversion

- [x] 5.1 Add `src/musubi_tuner/networks/lora_boogu_image.py` with Boogu-specific target module selection.
- [x] 5.2 Ensure LoRA creation skips deleted/unused projections in Boogu double-stream attention.
- [x] 5.3 Add `src/musubi_tuner/boogu_image/convert_lora_to_comfy.py` or equivalent conversion utility.
- [x] 5.4 Convert Musubi `lora_unet_*` keys to native `diffusion_model.*` Boogu keys.
- [x] 5.5 Preserve alpha-equivalent behavior in converted checkpoints.
- [x] 5.6 Add Boogu trainer post-save hook to emit converted checkpoints when enabled.
- [x] 5.7 Run the LoRA targeting and conversion tests.

## 6. Trainer Integration

- [x] 6.1 Add `BooguImageNetworkTrainer` in `boogu_image_train_network.py` inheriting from LTX `NetworkTrainer`.
- [x] 6.2 Implement Boogu model-specific args, bf16 defaults, unsupported direct torchao-fp8 guardrails, and optional safe fp8 handling.
- [x] 6.3 Implement Boogu transformer and VAE loading with no Qwen3-VL load during the training loop.
- [x] 6.4 Implement Boogu cached-batch preparation, instruction feature padding, and transformer call.
- [x] 6.5 Implement Musubi timestep to Boogu native time conversion and output sign/target handling.
- [x] 6.6 Add gradient checkpointing support through the Boogu block stacks.
- [x] 6.7 Add parser choices or script-level argument wiring required for Boogu training.
- [x] 6.8 Run parser/import/time-sign tests.

## 7. Snapshot Sampling

- [x] 7.1 Implement Boogu sample prompt processing that encodes Qwen3-VL sample embeddings up front and releases the text encoder.
- [x] 7.2 Implement Boogu native Euler snapshot sampler with resolution-aware time shift.
- [x] 7.3 Implement CFG sampling using positive and negative cached instruction features.
- [x] 7.4 Decode snapshot latents through the Boogu-compatible VAE and save images through the shared sample path.
- [x] 7.5 Add a CPU-level dry-run test around prompt parameter preparation where feasible without model weights.

## 8. fp8 And Memory Behavior

- [x] 8.1 Define the safe Boogu fp8 target/exclude rules from Boogu module names and the existing Krea2/Qwen fp8 patterns.
- [x] 8.2 Ensure fp8 mode keeps norms, embeddings, modulation, and output-sensitive modules in a safe dtype.
- [x] 8.3 Verify trainable LoRA overlays still operate on fp8-transformed base modules. Verified by the local `lucy-adamw8bit` run with `--fp8_base --fp8_scaled`: train LoRA forward/backward/optimizer completed, snapshot sampling applied `merged=83 runtime_overlays=418`, and step 100/200 checkpoints plus Comfy exports were produced under `/home/pyro/models/_out/boogu/lucy-adamw8bit`.
- [x] 8.4 Add clear errors/docs for unsupported direct HF `-fp8` torchao `.bin` training.

## 9. Docs And Local Configs

- [x] 9.1 Add `docs/boogu_image.md` documenting Base T2I scope, required weights, cache commands, train command, sampling, fp8 caveats, and unsupported Edit/Turbo scope.
- [x] 9.2 Add `.pyro/boogu/` configs and `train.sh` analogous to `.pyro/krea2`.
- [x] 9.3 Include a minimal one-step smoke config and a longer LoRA config template.
- [x] 9.4 Document expected local paths and how to adapt them for RunPod or another machine.

## 10. Verification

- [x] 10.1 Run the Boogu CPU test subset.
- [x] 10.2 Run existing Krea2/Qwen/Z-Image import or parser tests touched by shared glue.
- [x] 10.3 Run formatting/linting only if required by the files touched.
- [x] 10.4 Provide manual GPU smoke commands for latent cache, text cache, one-step train, first snapshot, and optional fp8 mode.
- [x] 10.5 Record known limitations and any deferred Boogu Edit/Turbo follow-up in implementation notes.
