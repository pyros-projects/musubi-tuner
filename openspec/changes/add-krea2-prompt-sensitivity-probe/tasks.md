## 1. Pure Helpers And Tests

- [x] 1.1 Add tests for collecting prompts from repeated `--prompt` values with `cli:<index>` provenance.
- [x] 1.2 Add tests for collecting non-empty prompt lines from `--prompt-file` with `<path>:<line>` provenance.
- [x] 1.3 Add tests for collecting `.txt` sidecars from `--sidecar-dir`, including deterministic sorting and optional recursive traversal.
- [x] 1.4 Add tests for report row sorting, fixed-width stdout table content, CSV export, and JSON export.
- [x] 1.5 Add tests for bypass projector patch extraction and application against a small fake `torch.nn.Linear(12, 1)`.
- [x] 1.6 Add tests for score calculation and coarse verdict thresholds using fake baseline/patched tensors.

## 2. Script Skeleton And CLI

- [x] 2.1 Create `scripts/krea2_prompt_sensitivity.py` with direct repo-root execution support for importing local `src/` modules.
- [x] 2.2 Implement argparse options for `--prompt`, `--prompt-file`, `--sidecar-dir`, `--recursive`, `--output`, `--format`, `--sort`, `--batch-size`, `--device`, `--dtype`, `--dit`, `--text_encoder`, `--bypass`, and `--bypass-strength`.
- [x] 2.3 Implement prompt collection helpers with source provenance and blank-prompt filtering.
- [x] 2.4 Implement stdout table rendering and CSV/JSON export without adding third-party dependencies.
- [x] 2.5 Implement clear validation errors for missing inputs, unsupported export formats, and missing path arguments.

## 3. Krea2 Fusion Scoring

- [x] 3.1 Implement partial Krea2 `txtfusion` construction from the existing Krea2 config.
- [x] 3.2 Implement partial DiT safetensors loading for `txtfusion.*` weights, accepting both native and `diffusion_model.`-prefixed key layouts.
- [x] 3.3 Implement bypass projector diff loading for `diffusion_model.txtfusion.projector.diff` and normalized variants.
- [x] 3.4 Implement prompt encoding through `load_krea2_text_encoder` and compaction through `gather_valid_text`.
- [x] 3.5 Implement baseline-vs-patched text-fusion forward passes using the same text attention parameter shape as Krea2.
- [x] 3.6 Implement numeric metrics, aggregate score, token count, and coarse verdict labels.
- [x] 3.7 Ensure the scoring path does not require a VAE path and does not invoke the Krea2 denoising sampler.
- [x] 3.8 Add absolute cosine redirect scoring that does not require baseline prompts.
- [x] 3.9 Keep per-batch GPU cleanup after scoring so prompt batches do not accumulate VRAM.

## 4. Verification And Operator Smoke Test

- [x] 4.1 Run the new unit tests for prompt collection, patching, scoring, and export behavior.
- [x] 4.2 Run existing Krea2 CPU tests to ensure the new script did not disturb Krea2 helpers.
- [x] 4.3 Run `python scripts/krea2_prompt_sensitivity.py --help` from the repository root.
- [x] 4.4 Run unit tests covering absolute redirect scoring and per-batch cleanup.
- [x] 4.5 Run `python scripts/krea2_prompt_sensitivity.py --help` from the repository root.
- [x] 4.6 Defer final real-model smoke to Pyro with the updated command.
