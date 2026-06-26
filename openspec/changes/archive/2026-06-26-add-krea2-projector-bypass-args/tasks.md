## 1. RED Tests

- [x] 1.1 Add CPU tests for loading supported Krea2 projector-bypass diff keys and fail clearly for missing keys.
- [x] 1.2 Add CPU tests for applying a bypass diff to a fake `txtfusion.projector.weight`, including weight scaling, dtype/device preservation, and shape mismatch errors.
- [x] 1.3 Add parser tests proving Krea2 training and standalone generation accept `--bypass` plus `--bypass-weight` and/or `--bypass_weight`.
- [x] 1.4 Add trainer tests proving active bypass metadata is emitted and inactive bypass metadata is omitted.
- [x] 1.5 Add a load-boundary test proving Krea2 training applies the bypass after DiT load and before returning the transformer.
- [x] 1.6 Add a standalone generation pipeline test proving `build_pipeline()` applies the bypass after loading the DiT and before generation setup.
- [x] 1.7 Run the targeted tests and verify they fail for the expected missing-bypass behavior before adding production code.

## 2. Core Implementation

- [x] 2.1 Implement a Krea2-local projector-bypass helper for loading supported safetensors keys and applying the weighted diff under `torch.no_grad()`.
- [x] 2.2 Add Krea2 training parser args for `--bypass`, `--bypass-weight`, and an underscore alias if needed.
- [x] 2.3 Apply the bypass inside `Krea2NetworkTrainer.load_transformer()` after `load_krea2_dit(...)` returns and before generic block-swap setup can run.
- [x] 2.4 Override `Krea2NetworkTrainer.get_checkpoint_metadata()` to record bypass path and weight only when active.
- [x] 2.5 Add standalone generation parser args for `--bypass`, `--bypass-weight`, and an underscore alias if needed.
- [x] 2.6 Apply the bypass inside `krea2_generate_image.build_pipeline()` after `load_krea2_dit(...)` returns and before eval/block-swap setup.

## 3. Local Experiment Surface

- [x] 3.1 Update `.pyro/krea2/train.sh` with optional `BYPASS` and `BYPASS_WEIGHT` environment controls.
- [x] 3.2 Ensure `.pyro/krea2/train.sh` appends bypass args only when `BYPASS` is non-empty.
- [x] 3.3 Update `.pyro/krea2/README.md` to document bypass usage, suggested consistency between training and inference, and the distinction from normal LoRA/base-weight merging.

## 4. GREEN Verification

- [x] 4.1 Run the targeted Krea2 bypass tests and verify they pass after implementation.
- [x] 4.2 Run the broader Krea2 CPU test subset, for example `uv run pytest tests/test_krea2_*.py -q`.
- [x] 4.3 Run lint/format checks for changed Python files, for example `uv run ruff check <changed files>`.
- [x] 4.4 Run `openspec validate add-krea2-projector-bypass-args --strict` or the repo's equivalent OpenSpec validation command.
- [x] 4.5 Defer GPU smoke tests unless the user confirms the GPU is free; provide exact training and generation smoke commands for later.
