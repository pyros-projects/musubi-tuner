## Why

Boogu-Image is a new Apache-2.0 open-weight image model with strong claims around dense text rendering, bilingual typography, and poster/product workflows, and ai-toolkit has already landed Boogu Image support upstream. `ltx-musubi` should gain a local, test-driven Boogu Base training path so we can evaluate LoRA quality against Krea2 without destabilizing the existing LTX/Krea2 work.

## What Changes

- Add Boogu Image Base text-to-image LoRA training support to the LTX Musubi branch.
- Add Boogu-specific transformer, attention, RoPE, Lumina-style block, and sampler helpers adapted from Boogu upstream and ai-toolkit's implementation.
- Add Boogu latent caching using the FLUX-compatible AutoencoderKL VAE layout released with Boogu.
- Add Boogu Qwen3-VL instruction-feature caching with natural-length per-caption embeddings and right-padding only at model-call time.
- Add Boogu snapshot sampling during training with the native Boogu flow-time convention, resolution-aware time shift, CFG support, and VAE decode.
- Add a Boogu LoRA network module that targets the model's transformer blocks without wrapping unused/deleted projections.
- Add Boogu LoRA save/load key conversion so checkpoints can be used in the native `diffusion_model.*` layout expected by Comfy-style tooling where possible.
- Add bf16-first training and optional fp8-scaled storage/compute support using the repo's existing safe quantization patterns; do not rely on the HF `-fp8` torchao `.bin` release for training in the initial change.
- Add docs and `.pyro/boogu` example configs/scripts analogous to the Krea2 workflow.
- Add CPU-first tests before implementation for parser/import behavior, cache metadata, instruction feature padding, timestep schedule, LoRA targets, key conversion, and save hooks.
- Keep Boogu Image Edit/TI2I, Turbo/DMD training, whole-model Turbo snapshot swapping, TeaCache/TaylorSeer inference caches, and trainer-student experiments out of this first change.

## Capabilities

### New Capabilities

- `boogu-image-training`: Boogu Image Base text-to-image LoRA training, caching, snapshot sampling, fp8-safe loading, and checkpoint conversion.

### Modified Capabilities

- None.

## Impact

- Adds Boogu modules, wrappers, docs, examples, and tests under `src/musubi_tuner`, top-level launch wrappers, `docs/`, `.pyro/boogu`, and `tests/`.
- Updates shared dataset/cache glue in `src/musubi_tuner/dataset/image_video_dataset.py`.
- Updates shared trainer/parser glue in `src/musubi_tuner/hv_train_network.py` only where Boogu needs architecture registration, timestep sampling, or save hooks.
- May require dependency validation for Qwen3-VL, Diffusers AutoencoderKL, and attention helpers already used by Krea2/Qwen paths.
- Uses ai-toolkit commit `60c1ac6` and the Boogu official repository/model card as implementation references, but ports only the training-relevant pieces needed for LTX Musubi.
