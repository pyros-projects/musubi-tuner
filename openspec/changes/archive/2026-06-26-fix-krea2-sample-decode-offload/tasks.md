## 1. Tests

- [x] 1.1 Add a CPU fake-object test proving Krea2 training decode offload happens after denoising and before the VAE moves to the sampling device.
- [x] 1.2 Add a CPU fake-object test proving Krea2 training decode offload leaves transformer state compatible with the shared sampling cleanup after successful and failed VAE decode.
- [x] 1.3 Add a standalone CLI/parser test for the Krea2 sampling offload flag.
- [x] 1.4 Add a standalone fake sampler test for offload-before-decode and restore-before-next-prompt behavior.

## 2. Training Snapshot Fix

- [x] 2.1 Add a small Krea2 helper/context manager for transformer decode offload with block-swap-aware movement.
- [x] 2.2 Wire the helper into `Krea2NetworkTrainer.do_inference` after denoising and before `vae.to(device)`.
- [x] 2.3 Make the helper respect `args.sample_with_offloading` and no-op when sampling offload is disabled.
- [x] 2.4 Preserve global sampling LoRA, prompt-local LoRA, and `--sample_blocks_to_swap` restore behavior.

## 3. Standalone Generation Fix

- [x] 3.1 Add a standalone Krea2 CLI flag for decode-time transformer offload, using Musubi's existing sampling-offload naming where practical.
- [x] 3.2 Pass the flag through `generate()` into the Krea2 sampling/decode path.
- [x] 3.3 Offload the DiT before VAE decode and restore it before the next prompt when continued generation needs it.
- [x] 3.4 Keep current resident-DiT behavior unchanged when the flag is unset.

## 4. Verification

- [x] 4.1 Run the new CPU tests.
- [x] 4.2 Run existing Krea2 CPU tests that cover sampling LoRA, Comfy conversion, and prompt sensitivity script behavior.
- [x] 4.3 Run Krea2 generation help output to confirm the new flag is visible.
- [x] 4.4 Defer real GPU smoke until the GPU is free and document the smoke command in the implementation final.
