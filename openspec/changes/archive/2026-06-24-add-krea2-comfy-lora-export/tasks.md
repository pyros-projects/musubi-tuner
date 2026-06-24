## 1. Converter

- [x] 1.1 Add a Krea2 LoRA-to-Comfy converter module or script that reads Musubi `lora_unet_*` Krea2 LoRA safetensors and writes native `diffusion_model.*` Krea2 keys.
- [x] 1.2 Implement explicit Krea2 module-name restoration so `layerwise_blocks`, `refiner_blocks`, `txtmlp`, `tmlp`, `tproj`, `last.linear`, and main `blocks.*` paths convert correctly.
- [x] 1.3 Preserve safetensors metadata during conversion.
- [x] 1.4 Drop `.alpha` keys in Comfy output and fold non-default `alpha/rank` scale into the up weight.

## 2. Training Export Hook

- [x] 2.1 Add `Krea2NetworkTrainer.post_save_checkpoint_hook()` that writes `<checkpoint>.comfy.safetensors` when `convert_to_comfy` is enabled.
- [x] 2.2 Reuse the existing shared flag behavior for `--no_convert_to_comfy`, `--save_original_lora`, and `--no_save_original_lora`.
- [x] 2.3 Upload the Comfy companion checkpoint when `huggingface_repo_id` is configured, matching LTX2/Z-Image behavior.
- [x] 2.4 Ensure checkpoint rotation continues to remove stale `.comfy.safetensors` companions through the existing shared trainer cleanup path.

## 3. Tests

- [x] 3.1 Add converter tests for main Krea2 blocks, `txtfusion.layerwise_blocks`, `txtfusion.refiner_blocks`, `txtfusion.projector`, `tmlp`, `txtmlp`, `tproj`, `first`, and `last.linear`.
- [x] 3.2 Add tests proving no converted key contains corrupted `txtfusion.layerwise.blocks` or `txtfusion.refiner.blocks` paths.
- [x] 3.3 Add alpha tests for rank-equal alpha and non-default alpha folding.
- [x] 3.4 Add Krea2 post-save hook tests for default dual-save behavior, disabled conversion, metadata preservation, and `--no_save_original_lora` removal.
- [x] 3.5 Run focused Krea2/LTX converter tests without requiring GPU or full model weights.

## 4. Docs

- [x] 4.1 Update `docs/krea2.md` to describe default dual-save behavior and the `.comfy.safetensors` artifact.
- [x] 4.2 Document that training resume requires the original non-Comfy checkpoint and that `--no_save_original_lora` should not be used if resume from that checkpoint is needed.
- [x] 4.3 Add a standalone converter usage example for existing Krea2 checkpoints.
