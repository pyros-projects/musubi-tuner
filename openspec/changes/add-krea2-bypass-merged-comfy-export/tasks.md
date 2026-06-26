## 1. Test First

- [x] 1.1 Add failing tests for bypass weight filename tokenization, including `5 -> w5`, `0.5 -> w0p5`, and `-1 -> wm1`.
- [x] 1.2 Add failing converter tests proving a bypass-merged Comfy state dict/file includes the scaled `diffusion_model.txtfusion.projector.diff` tensor and preserves normal LoRA tensors.
- [x] 1.3 Add failing converter tests proving the normal `.comfy.safetensors` conversion remains unchanged and does not include bypass diff tensors.
- [x] 1.4 Add failing metadata tests proving the bypass-merged file preserves source metadata and records bypass source path, weight, and merge status.
- [x] 1.5 Add failing parser/validation tests proving `--bypass-merge` is accepted, requires `--bypass`, and conflicts with `--no_convert_to_comfy`.
- [x] 1.6 Add failing post-save hook tests proving Krea2 training writes both `.comfy.safetensors` and `.comfy.bypassed.<weight>.safetensors` when bypass merge is active.
- [x] 1.7 Add failing local script tests or shell assertions proving `.pyro/krea2/train.sh` passes `--bypass-merge` only when the corresponding environment flag is enabled.

## 2. Converter Implementation

- [x] 2.1 Add a Krea2 converter helper for formatting safe bypass weight filename tokens.
- [x] 2.2 Extend the Krea2 Comfy converter with an optional bypass-merged output path that loads the bypass diff through the existing projector-bypass helper.
- [x] 2.3 Store the scaled bypass diff under `diffusion_model.txtfusion.projector.diff` in the bypass-merged output.
- [x] 2.4 Preserve source metadata in the bypass-merged output and add explicit bypass merge metadata.
- [x] 2.5 Keep existing `convert_lora_to_comfy()` behavior backward-compatible for callers that do not request bypass merge.

## 3. Training Surface

- [x] 3.1 Add `--bypass-merge` and underscore alias if needed to the Krea2 training parser.
- [x] 3.2 Validate that `--bypass-merge` requires `--bypass` before training starts.
- [x] 3.3 Validate that `--bypass-merge` conflicts with `--no_convert_to_comfy` before training starts.
- [x] 3.4 Update `Krea2NetworkTrainer.post_save_checkpoint_hook()` to write the bypass-merged Comfy companion after the normal Comfy export and before optional original-checkpoint deletion.
- [x] 3.5 Ensure Hugging Face upload behavior, if active, uploads the bypass-merged companion consistently with the normal Comfy companion.

## 4. Local Operator Surface

- [x] 4.1 Add an opt-in `.pyro/krea2/train.sh` environment flag such as `BYPASS_MERGE=1`.
- [x] 4.2 Make `.pyro/krea2/train.sh` pass `--bypass-merge` only when merge is enabled and `BYPASS` is non-empty.
- [x] 4.3 Update `.pyro/krea2/README.md` to document the merged artifact name, load-at-1.0 guidance, and global-strength coupling caveat.

## 5. Verification

- [x] 5.1 Run the newly added focused tests and verify they fail before implementation for the expected missing behavior.
- [x] 5.2 Run the focused Krea2 Comfy export/projector bypass tests after implementation.
- [x] 5.3 Run `uv run pytest tests/test_krea2_*.py -q`.
- [x] 5.4 Run `uv run ruff check` on all changed Python files and tests.
- [x] 5.5 Run `bash -n .pyro/krea2/train.sh`.
- [x] 5.6 Run `openspec validate add-krea2-bypass-merged-comfy-export --strict`.
- [x] 5.7 Provide a small existing-checkpoint smoke command that converts one Krea2 LoRA plus bypass into a `.comfy.bypassed.<weight>.safetensors` artifact without starting a GPU training run, if a standalone converter path is implemented.
