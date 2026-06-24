## Why

Krea 2 support has landed upstream in Musubi, but `ltx-musubi` is a divergent LTX-focused fork with local trainer, sampling-LoRA, and dataset changes that make a direct upstream merge risky. We need a manual port that brings over Krea 2 RAW/base LoRA training while preserving the fork's existing LTX work and Pyro's Turbo-LoRA snapshot workflow.

## What Changes

- Add Krea 2 image training support to `ltx-musubi`, including DiT loading, Qwen3-VL text embedding caches, Qwen-Image VAE latent caches, LoRA training, and snapshot sampling.
- Add Krea 2 scaled-fp8 support matching upstream's intent: require `--fp8_base --fp8_scaled`, quantize the main `blocks.*` Linear weights, and keep modulation, norms, and text-fusion in bf16.
- Integrate Krea 2 with the existing LTX sampling-LoRA adapter path so training uses the RAW/base model while snapshots can apply the Turbo LoRA as a temporary sampling adapter.
- Add Krea 2 sampling-LoRA key normalization for local Turbo LoRA files that use `diffusion_model.*` native keys.
- Add Krea 2 timestep and snapshot schedule support, including `krea2_shift` for training and a `mu=1.15` Turbo-preview override for snapshot sampling.
- Port only the shared glue required by Krea 2 into the older LTX layout; do not replace the LTX fork's trainer or dataset architecture with upstream's newer split modules.
- Keep upstream whole-DiT `--turbo_dit` swapping out of the initial scope. The supported preview path is Turbo LoRA as a sampling adapter, not a full Turbo base checkpoint swap.
- Bump or otherwise handle the Transformers dependency required for Qwen3-VL text encoding.

## Capabilities

### New Capabilities

- `krea2-training`: Krea 2 RAW/base LoRA training, caching, scaled-fp8 loading, and in-training snapshot sampling with optional Turbo-LoRA sampling adapter.

### Modified Capabilities

- None.

## Impact

- Adds Krea 2 model, training, cache, inference, LoRA, docs, and tests under `src/musubi_tuner`, top-level wrappers, `docs/`, and `tests/`.
- Updates shared LTX fork glue in `src/musubi_tuner/dataset/image_video_dataset.py`, `src/musubi_tuner/hv_train_network.py`, `src/musubi_tuner/modules/attention.py`, and `src/musubi_tuner/utils/sai_model_spec.py`.
- Updates dependency metadata for Qwen3-VL support, most likely `transformers==4.57.6`.
- Must preserve existing dirty/local LTX changes, including prompt-local sampling LoRA work and compile prewarm tests.
