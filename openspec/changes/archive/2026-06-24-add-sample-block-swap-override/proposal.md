## Why

Krea2 training can benefit from `--blocks_to_swap` because it makes larger real batches possible, but snapshot sampling becomes much slower because every denoise step streams swapped blocks between CPU and GPU. Sampling is no-grad and often has enough VRAM to run with fewer swapped blocks than training, so users need a sampling-only override.

## What Changes

- Add a sampling-time block-swap override for Krea2 training snapshots.
- Allow users to inherit the training swap count, disable block swap only during sampling, or use a smaller sampling swap count.
- Restore the original training block-swap state after sampling even if sampling fails.
- Keep the existing `--blocks_to_swap` training behavior unchanged.
- Document when to use the override and the expected memory/speed tradeoff.

## Capabilities

### New Capabilities

### Modified Capabilities
- `krea2-training`: Krea2 snapshot sampling can temporarily override train-time block swap and restore the training state afterward.

## Impact

- Affects `src/musubi_tuner/hv_train_network.py` sampling orchestration and parser options.
- Affects `src/musubi_tuner/krea2/krea2_mmdit.py` block-swap state management.
- May add CPU-testable coverage for block-swap state override/restore behavior without requiring GPU or model weights.
- Updates `docs/krea2.md` with sampling override guidance.
