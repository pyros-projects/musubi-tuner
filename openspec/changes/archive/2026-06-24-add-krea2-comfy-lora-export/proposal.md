## Why

Krea2 LoRA training currently saves only the Musubi `lora_unet_*` checkpoint, while LTX2 and Z-Image already produce a `.comfy.safetensors` companion by default. Krea2 operators should get the same direct ComfyUI artifact so trained LoRAs can be used without a manual, error-prone conversion step.

## What Changes

- Add a Krea2 LoRA-to-Comfy converter for Musubi `lora_unet_*` Krea2 keys.
- Convert Krea2 saved LoRA checkpoints to native Krea2 Comfy key layout by default, writing `<name>.comfy.safetensors` next to the original checkpoint.
- Preserve the existing global save controls:
  - default: save both original and Comfy files
  - `--no_convert_to_comfy`: save only the original
  - `--no_save_original_lora`: keep only the Comfy file after conversion
- Preserve metadata in converted files and keep checkpoint rotation/upload behavior aligned with the existing shared trainer/LTX2 hooks.
- Document that training resume must use the original non-Comfy checkpoint format.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `krea2-training`: add automatic Krea2 ComfyUI LoRA export and a standalone converter for trained Krea2 LoRA checkpoints.

## Impact

- Affected code: `src/musubi_tuner/krea2_train_network.py`, a new Krea2 converter module or script, Krea2 tests, and Krea2 docs.
- Affected operator surface: Krea2 training will honor the already-existing `--no_convert_to_comfy`, `--save_original_lora`, and `--no_save_original_lora` flags.
- Compatibility: original Musubi checkpoints remain available by default for resume/training workflows; converted Comfy checkpoints use native `diffusion_model.*` Krea2 paths suitable for ComfyUI loading.
- Risk: a generic underscore-to-dot converter would corrupt names such as `txtfusion.layerwise_blocks` and `txtfusion.refiner_blocks`, so Krea2 needs explicit conversion rules and tests.
