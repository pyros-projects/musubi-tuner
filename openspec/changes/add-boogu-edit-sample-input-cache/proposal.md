## Why

Boogu Edit LoRA runs can already produce useful output-only training for cases where the base model's editing priors are enough, but current snapshot sampling cannot prove edit behavior because sample prompts do not attach source images. Encoding source images during each sample would also stack VAE/image preprocessing work onto the transformer sampling window, which is the most memory-sensitive part of the run.

## What Changes

- Add optional source-image support for Boogu snapshot sample prompts, with a global prompt default and per-prompt override.
- Pre-cache Boogu Edit sample inputs before the training run by encoding both the Qwen3-VL image-conditioned instruction features and the VAE reference image latents.
- Keep Boogu training batches output-only; input images affect snapshot sampling only.
- Preserve existing no-image sample prompt behavior for text-to-image style Boogu samples.
- Validate sample input image paths up front and make unsupported multi-image inputs explicit rather than silently ambiguous.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `boogu-image-training`: Extend Boogu snapshot sampling so sample prompts can carry optional pre-cached edit input images.

## Impact

- Affected code: `src/musubi_tuner/boogu_image_train_network.py`, Boogu prompt/text-encoder helper code, and focused Boogu sampling tests.
- Affected configuration: Boogu sample prompt TOML can use a canonical single-image key for global or per-prompt edit inputs, with compatibility aliases considered during design.
- Runtime behavior: Boogu sample input images are encoded before the transformer is loaded for training/sampling, then reused during snapshot generation.
- Dependencies: No new external runtime dependency is expected.
