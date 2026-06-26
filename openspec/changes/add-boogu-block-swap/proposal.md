## Why

Boogu training currently rejects `--blocks_to_swap`, even though the model is large enough that block swap is a practical low-VRAM requirement. The transformer now exposes no-op block-swap methods only to satisfy shared sampler calls; those stubs should be replaced with real `ModelOffloader` support using the same lifecycle as other Musubi model families.

## What Changes

- Implement real Boogu transformer block swap using the existing `ModelOffloader` infrastructure.
- Manage Boogu blocks in forward execution order across context, noise, reference-image, double-stream, and single-stream stages.
- Remove the Boogu trainer guard that rejects positive `--blocks_to_swap`.
- Add Boogu support for `--sample_blocks_to_swap` through the shared sampling override contract.
- Preserve the current no-op behavior when block swap is not enabled.
- Add CPU-first tests for setup validation, device movement, forward wait/submit ordering, inference/training switches, and sampling overrides.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `boogu-image-training`: Boogu Image training supports `--blocks_to_swap` and `--sample_blocks_to_swap` using the repository's shared block-swap lifecycle.

## Impact

- Updates `src/musubi_tuner/boogu_image/transformer.py` to replace no-op block-swap methods with real offloader wiring.
- Updates `src/musubi_tuner/boogu_image_train_network.py` to accept positive `--blocks_to_swap`.
- Extends Boogu CPU coverage in `tests/test_boogu_image_support.py` or a dedicated block-swap test file.
- Does not change Boogu model math, sample schedule, LoRA targeting, fp8 rules, or checkpoint format.
