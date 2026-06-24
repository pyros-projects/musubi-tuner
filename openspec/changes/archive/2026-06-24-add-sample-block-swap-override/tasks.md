## 1. Parser And Configuration

- [x] 1.1 Add shared trainer parser option `--sample_blocks_to_swap` with default `None` so missing config inherits train-time `--blocks_to_swap`.
- [x] 1.2 Ensure `0` is preserved as an explicit value and is not collapsed into the unset/default path.
- [x] 1.3 Include clear help text describing inherit, `0` unswapped sampling, and positive sampling override behavior.

## 2. Krea2 Block-Swap Override

- [x] 2.1 Add a Krea2 transformer helper/context manager for temporarily overriding block-swap count during sampling.
- [x] 2.2 Make the helper wait for pending offloader transfers before changing or restoring block-swap state.
- [x] 2.3 Update both model-level `blocks_to_swap` and offloader-level `blocks_to_swap` during override and restore.
- [x] 2.4 Rebuild the block device layout for the requested sampling count, including moving all blocks to accelerator when the count is `0`.
- [x] 2.5 Validate invalid counts and positive overrides without an initialized offloader with clear errors.

## 3. Sampling Integration

- [x] 3.1 Wire `hv_train_network.sample_images()` to enter the model override only when `--sample_blocks_to_swap` is explicitly set and the transformer supports it.
- [x] 3.2 Preserve current sampling behavior when `--sample_blocks_to_swap` is unset or unsupported by the current architecture.
- [x] 3.3 Ensure the override restoration runs before the existing training-mode block-swap restoration path.
- [x] 3.4 Log the effective sampling block-swap count when an override is active.

## 4. Tests

- [x] 4.1 Add CPU tests for Krea2 override `None`/inherit no-op behavior.
- [x] 4.2 Add CPU tests for temporary `0` override moving all blocks to the requested device layout and restoring the original count.
- [x] 4.3 Add CPU tests for a positive sampling override and restoration.
- [x] 4.4 Add CPU tests proving restoration runs when sampling raises after entering the override.
- [x] 4.5 Add CPU tests for invalid override requests and explicit `0` parser/config handling.

## 5. Docs And Examples

- [x] 5.1 Update `docs/krea2.md` to document `--sample_blocks_to_swap`.
- [x] 5.2 Explain the recommended Krea2 usage: train with `--blocks_to_swap N`, sample with `--sample_blocks_to_swap 0` or a smaller positive count when VRAM allows.
- [x] 5.3 Document the OOM tradeoff and fallback to inherited block swap.

## 6. Verification

- [x] 6.1 Run focused CPU tests covering Krea2 block-swap override behavior.
- [x] 6.2 Run existing focused Krea2 sampling tests to guard snapshot-sampling regressions.
- [x] 6.3 Run `openspec validate add-sample-block-swap-override --strict`.
