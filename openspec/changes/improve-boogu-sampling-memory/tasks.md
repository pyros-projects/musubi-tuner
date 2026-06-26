## 1. TDD Coverage

- [ ] 1.1 Add a CPU test showing Boogu `--sample_with_offloading` moves the transformer to CPU before VAE decode.
- [ ] 1.2 Add a CPU test showing Boogu sampling without `--sample_with_offloading` does not perform the decode-time transformer offload.
- [ ] 1.3 Add a CPU test showing VAE cleanup still runs when Boogu decode raises.
- [ ] 1.4 Add a CPU test showing the Boogu denoise loop wraps timestep transitions in a progress iterator with the expected total.

## 2. Shared Decode-Offload Helper

- [ ] 2.1 Extract Krea2's transformer decode-offload helper into a shared utility or keep an equivalent helper available to Boogu with the same tested contract.
- [ ] 2.2 Update Krea2 imports if the helper is extracted and keep the existing Krea2 decode-offload behavior unchanged.
- [ ] 2.3 Ensure the helper synchronizes pending block-swap transfers when the transformer exposes `_wait_for_pending_block_swaps`.

## 3. Boogu Sampling Updates

- [ ] 3.1 Wrap Boogu VAE decode with the decode-offload helper when `args.sample_with_offloading` is enabled.
- [ ] 3.2 Keep VAE `to(device)`, eval, decode, CPU restore, and memory cleanup behavior intact around the new helper.
- [ ] 3.3 Add a progress bar around the Boogu native timestep transition loop using the repository's existing `tqdm` style.

## 4. Verification

- [ ] 4.1 Run `uv run pytest tests/test_boogu_image_support.py tests/test_krea2_decode_offload.py -q`.
- [ ] 4.2 Run `openspec validate improve-boogu-sampling-memory --strict`.
- [ ] 4.3 Record that real GPU smoke sampling remains deferred until the GPU is free.
