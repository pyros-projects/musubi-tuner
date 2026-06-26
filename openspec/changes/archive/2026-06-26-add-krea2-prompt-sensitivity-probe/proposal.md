## Why

Krea2 prompt behavior can change sharply when the local filter-bypass projector patch is applied, especially for captions with ambiguous or loaded body-language semantics. Operators need a lightweight way to scan prompts and sidecar captions before training so they can spot bypass-sensitive captions without running full image sampling.

## What Changes

- Add a standalone Krea2 prompt-sensitivity probe script under `scripts/`.
- Support prompt input from repeated CLI arguments, a one-prompt-per-line text file, and a folder of `.txt` sidecar captions.
- Print a ranked table with prompt provenance, token count, numeric sensitivity metrics, and a coarse verdict.
- Export the same report to CSV or JSON for dataset review.
- Compare baseline Krea2 text-fusion conditioning against conditioning with a direct bypass projector patch applied.
- Keep the default probe lightweight by loading the Krea2 text encoder and text-fusion weights only, not the VAE or full denoising sampler.

## Capabilities

### New Capabilities

- `krea2-prompt-sensitivity-probe`: Covers the standalone script and report contract for probing prompt sensitivity to a Krea2 bypass projector patch.

### Modified Capabilities

- None.

## Impact

- Adds `scripts/krea2_prompt_sensitivity.py`.
- Adds CPU-testable helpers or tests for prompt collection, report formatting, bypass patch application, and score calculation.
- Reuses existing Krea2 text encoder, prompt compaction, and text-fusion modules from `src/musubi_tuner/krea2`.
- Requires local Krea2 DiT and Qwen3-VL text encoder paths at runtime for real scoring.
