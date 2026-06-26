## Why

Boogu snapshot sampling now works, but the denoise loop is silent and `--sample_with_offloading` does not protect the VAE decode phase from stacking the VAE on top of the transformer. Krea2 already solved the same memory pattern, so Boogu should gain equivalent sampling visibility and decode-time offload before longer runs make this painful to debug.

## What Changes

- Add a progress indicator around the Boogu native snapshot denoise loop.
- Make Boogu snapshot sampling honor `--sample_with_offloading` through VAE decode by moving the transformer off the sampling device before the VAE is loaded for decoding.
- Reuse or extract the existing Krea2 decode-offload helper so pending block-swap transfers are synchronized before cleanup and decode.
- Preserve current sampling behavior when `--sample_with_offloading` is disabled.
- Add CPU-first tests for offload ordering, failure restoration, and progress loop wiring.
- Keep Boogu block-swap implementation out of this change; that is tracked separately.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `boogu-image-training`: Boogu snapshot sampling reports denoise progress and avoids VAE/transformer memory stacking during decode when sample offloading is enabled.

## Impact

- Updates Boogu snapshot sampling in `src/musubi_tuner/boogu_image_train_network.py`.
- May extract shared decode-offload logic from `src/musubi_tuner/krea2/krea2_sampling.py` into a small shared utility used by both Krea2 and Boogu.
- Adds or extends CPU tests in `tests/test_boogu_image_support.py` and, if a shared helper is extracted, keeps `tests/test_krea2_decode_offload.py` passing.
- No new model weights, CLI flags, or external dependencies are required.
