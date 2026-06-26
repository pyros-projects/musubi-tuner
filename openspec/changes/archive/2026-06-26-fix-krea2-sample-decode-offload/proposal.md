## Why

Krea2 snapshot sampling currently moves the Qwen-Image VAE onto the GPU while the Krea2 transformer is still resident, so decode can use more VRAM than denoising. LTX2 already has a staged offload path before final VAE decode; Krea2 should honor the same intent when sampling offload is requested.

## What Changes

- Fix Krea2 in-training snapshot sampling so `--sample_with_offloading` offloads the transformer before VAE decode, not only after the whole sample returns.
- Add an equivalent standalone Krea2 generation option so `krea2_generate_image.py` can offload the DiT before decode when requested.
- Restore the transformer to the expected sampling/training device state after decode, including block-swap state where applicable.
- Keep current behavior unchanged when sampling offload is not requested.
- Add CPU-testable coverage using fake transformer/VAE objects to verify call ordering and restoration without loading real model weights.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `krea2-training`: Krea2 snapshot sampling and standalone inference shall support decode-time transformer offload before moving the VAE to GPU.

## Impact

- Affects Krea2 snapshot sampling in `src/musubi_tuner/krea2_train_network.py`.
- Affects standalone Krea2 image generation in `src/musubi_tuner/krea2_generate_image.py` and/or `src/musubi_tuner/krea2/krea2_sampling.py`.
- May reuse existing `--sample_with_offloading` for training snapshots and add a standalone CLI flag with matching semantics.
- Requires focused CPU tests for offload/decode/restore ordering; GPU smoke can remain manual.
