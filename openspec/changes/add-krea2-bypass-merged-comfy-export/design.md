## Context

Krea2 training now supports `--bypass <path> --bypass-weight <x>` by applying a direct `txtfusion.projector` diff to the loaded DiT before training and snapshots. Krea2 training also exports each saved LoRA to a native ComfyUI companion file through `Krea2NetworkTrainer.post_save_checkpoint_hook()` and `musubi_tuner.krea2.convert_lora_to_comfy`.

When a LoRA is trained with an active bypass, the most faithful end-user inference setup is `base + bypass at x + trained LoRA`. ComfyUI can load the small bypass file through normal LoRA tools, but shipping two files makes the trained LoRA easier to misapply. The desired workflow is an optional additional ComfyUI artifact that bundles the scaled bypass tensor into the exported LoRA file.

## Goals / Non-Goals

**Goals:**

- Add an opt-in Krea2 training argument, `--bypass-merge`, that emits a bypass-merged ComfyUI checkpoint on every Krea2 LoRA save when `--bypass` is active.
- Keep the existing original Musubi checkpoint and normal `.comfy.safetensors` export behavior unchanged.
- Encode the bypass weight in the filename, for example `.comfy.bypassed.w5.safetensors`.
- Store the bypass projector diff already scaled by `--bypass-weight`, so loading the merged artifact at strength `1.0` matches the training-time bypass.
- Preserve metadata and add enough bypass-merge metadata to inspect how the artifact was produced.
- Add focused CPU tests for conversion, parser validation, filename tokenization, and post-save behavior.

**Non-Goals:**

- Do not change how training applies the bypass to the live model.
- Do not merge the bypass into the base DiT checkpoint.
- Do not attempt to make the embedded bypass independent from the ComfyUI global LoRA strength; that would require loader-specific behavior outside this repo.
- Do not remove the normal `.comfy.safetensors` companion file.

## Decisions

1. **Emit an additional artifact rather than changing `.comfy.safetensors`.**

   The normal Comfy export remains a pure trained LoRA. The bypass-merged export gets an explicit filename marker. This avoids surprising operators who want to compare with and without the bypass or who already have pipelines expecting the current `.comfy.safetensors` file.

2. **Scale the bypass diff into the saved tensor.**

   The merged file should be loaded at strength `1.0` to reproduce `base + trained_lora + bypass_weight * bypass_diff`. Therefore the stored `diffusion_model.txtfusion.projector.diff` tensor is `raw_diff * bypass_weight`. The metadata records the source path and numeric weight.

3. **Use a safe deterministic weight token.**

   The filename token is derived from Python `:g` formatting and made filesystem-safe: `5` becomes `w5`, `0.5` becomes `w0p5`, and `-1` becomes `wm1`. This keeps the common `w5` form compact while avoiding dots in the middle of the semantic filename segment.

4. **Validate contradictory args early.**

   `--bypass-merge` requires `--bypass` and conflicts with `--no_convert_to_comfy`. Failing early is better than silently not producing the file the operator asked for.

5. **Implement through the Krea2 converter layer.**

   The converter already owns native Comfy key layout and metadata preservation. Add a small helper or converter option there to append the scaled bypass diff and write the bypassed output. The trainer post-save hook should only decide when to call it.

## Risks / Trade-offs

- **Risk: Global LoRA strength also scales the embedded bypass in ComfyUI.** -> Document this clearly in `.pyro/krea2/README.md`; the merged artifact is intended for load strength `1.0`.
- **Risk: Unsupported bypass key variants drift from training-time support.** -> Reuse `projector_bypass.SUPPORTED_PROJECTOR_DIFF_KEYS` or `load_projector_bypass_diff()` instead of duplicating key lookup.
- **Risk: `--no_save_original_lora` deletion could remove the source before all exports finish.** -> Generate both normal and bypass-merged Comfy files before deleting the original Musubi checkpoint.
- **Risk: Filename weight formatting could collide for odd floats.** -> Use a single helper with tests; if two textual weights normalize to the same numeric value, sharing an output name is acceptable because they represent the same effective bypass weight.
