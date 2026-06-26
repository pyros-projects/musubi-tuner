## Context

Krea2 support currently has two LoRA-related mechanisms:

- `--base_weights` merges normal LoRA/network weights into the DiT before training.
- `--sampling_lora_weight` temporarily applies normal LoRA weights for in-training snapshot sampling.

The filter-bypass artifact used for Krea2 is different: it is a direct projector patch such as `diffusion_model.txtfusion.projector.diff` with shape `(1, 12)`. It should be applied to `dit.txtfusion.projector.weight`, not converted into `lora_unet_*` rank-pair modules.

Krea2 training loads the DiT through `Krea2NetworkTrainer.load_transformer()` before generic block swap is enabled. Standalone generation loads the DiT in `krea2_generate_image.build_pipeline()` before eval/block-swap setup. Those are the safest insertion points.

## Goals / Non-Goals

**Goals:**

- Provide explicit Krea2 bypass CLI args for training and standalone generation.
- Apply a supported projector diff exactly once to the loaded Krea2 DiT before training or inference uses it.
- Make in-training snapshot sampling automatically use the same patched transformer as training.
- Preserve default behavior when no bypass path is provided.
- Add CPU-testable helper coverage and parser/metadata tests before production code.

**Non-Goals:**

- Do not make bypass diffs part of the generic LoRA or `--base_weights` system.
- Do not train the bypass patch itself.
- Do not infer or auto-tune bypass weight from prompt sensitivity scores.
- Do not add prompt-local bypass overrides in this change.
- Do not bake the bypass patch into saved LoRA weights; the output LoRA remains a LoRA trained against the operator's configured base condition.

## Decisions

### Add a small Krea2 projector-bypass helper

Create a Krea2-local helper, for example `musubi_tuner.krea2.projector_bypass`, with pure functions:

- `load_projector_bypass_diff(path) -> torch.Tensor`
- `apply_projector_bypass(dit, diff, weight, source_path=None) -> None`

The loader accepts the known direct-diff key variants:

- `diffusion_model.txtfusion.projector.diff`
- `txtfusion.projector.diff`
- `diffusion_model.txtfusion.projector.weight.diff`
- `txtfusion.projector.weight.diff`

The applier validates that `dit.txtfusion.projector.weight` exists and has the same shape as the diff, then applies `projector.weight += diff * weight` under `torch.no_grad()`. It should preserve the projector weight device and dtype by converting only the diff tensor at application time.

Alternative considered: convert the diff into a fake LoRA or `--base_weights` input. Rejected because the checkpoint has no rank-pair structure and the generic LoRA path would create misleading failure modes.

### Apply at Krea2 DiT load boundaries

Training should call the helper inside `Krea2NetworkTrainer.load_transformer()` after `krea2_utils.load_krea2_dit(...)` returns and before the model is returned to generic trainer setup. This means block-swap setup, train-time forward, and snapshot sampling all see the same patched transformer.

Standalone generation should call the same helper inside `krea2_generate_image.build_pipeline()` after `load_krea2_dit(...)` returns and before eval/block-swap setup. This keeps inference behavior aligned with training.

Alternative considered: apply during sampling only. Rejected because training gradients would still be computed against the unpatched conditioning path, which is exactly the mismatch this change is meant to avoid.

### CLI names

Add user-facing args:

- `--bypass <path>`
- `--bypass-weight <float>`

If parser consistency benefits from it, also accept `--bypass_weight` as an alias for the same destination. The CLI default is no bypass. When a bypass path is supplied and no weight is given, default to `1.0`; local scripts can default to `5` for the known Comfy-style bypass artifact.

Alternative considered: use `--base_weights`. Rejected because it would imply normal LoRA merge semantics.

### Metadata

Override `Krea2NetworkTrainer.get_checkpoint_metadata()` to include bypass metadata when active, such as:

- `ss_krea2_bypass_path`
- `ss_krea2_bypass_weight`

This does not make the LoRA self-applying. It gives operators and future scripts a breadcrumb that the LoRA was trained against a patched Krea2 projector.

### Local training script surface

Update `.pyro/krea2/train.sh` with optional environment controls:

- `BYPASS=""`
- `BYPASS_WEIGHT="5"`

The script should append the bypass args only when `BYPASS` is non-empty. The README should document that LoRAs trained with a bypass should normally be sampled/inferred with the same bypass and weight.

## Risks / Trade-offs

- Projector diff shape changes in future bypass artifacts -> fail early with a clear shape/key error rather than silently applying the wrong tensor.
- Applying a high bypass weight can alter otherwise stable prompts -> keep the weight explicit and record it in metadata.
- LoRAs trained with bypass may behave differently without the same bypass at inference -> document the consistency requirement and write metadata.
- In-place patch is not reversible without reload -> acceptable because the bypass is a process-level configuration applied once at load time.

## Migration Plan

No migration is required. Existing training and generation commands remain unchanged when `--bypass` is omitted. Rollback is simply removing the bypass args or setting the local `BYPASS` environment variable to empty.

## Open Questions

- None for implementation. If later experiments need prompt-local bypass weights, that should be a separate change.
