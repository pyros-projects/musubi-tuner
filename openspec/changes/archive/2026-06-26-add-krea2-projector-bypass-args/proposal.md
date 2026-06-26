## Why

Krea2 filter-bypass checkpoints can be direct `txtfusion.projector` diffs rather than normal LoRA rank pairs, so routing them through `--base_weights`, `--network_weights`, or sampling LoRA plumbing is the wrong abstraction. Operators need a clear way to apply the same projector bypass during training, in-training snapshots, and standalone inference so concepts affected by the default Krea2 conditioning can be trained and previewed on the same conditioning surface.

## What Changes

- Add Krea2-only bypass CLI options for training and standalone generation:
  - `--bypass <path>`
  - `--bypass-weight <float>` with `--bypass_weight` as a parser alias if useful for Musubi consistency
- Load supported safetensors projector-diff keys such as `diffusion_model.txtfusion.projector.diff` and apply `diff * weight` to the loaded Krea2 DiT `txtfusion.projector.weight`.
- Apply the bypass once after Krea2 DiT load and before training/sampling uses the transformer, so in-training snapshots automatically see the same patched transformer as training.
- Record bypass metadata in saved Krea2 LoRA checkpoints so later operators can tell that a LoRA was trained against a patched Krea2 projector.
- Update the local `.pyro/krea2/train.sh` and README surface so experiments can enable the bypass without hand-editing commands.
- Do not treat bypass diffs as trainable LoRAs, sampling LoRAs, or `--base_weights` merge inputs.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `krea2-training`: Krea2 training, snapshot sampling, and standalone inference can apply an explicit projector bypass diff with a configured weight.

## Impact

- Affected code:
  - `src/musubi_tuner/krea2_train_network.py`
  - `src/musubi_tuner/krea2_generate_image.py`
  - likely a small shared Krea2 bypass helper under `src/musubi_tuner/krea2/`
  - Krea2 unit tests under `tests/`
  - `.pyro/krea2/train.sh`
  - `.pyro/krea2/README.md`
- No new runtime dependency is expected; safetensors and torch are already used.
- No breaking changes: default behavior remains unchanged when `--bypass` is omitted.
