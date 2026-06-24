## Context

Krea2 training currently uses `networks.lora_krea2`, which saves the training checkpoint in Musubi/Kohya-style `lora_unet_*` format. That format is useful for resume and internal sampling, but it is not the native Krea2 ComfyUI layout used by the local Turbo LoRA reference (`diffusion_model.*.lora_down.weight` / `lora_up.weight`).

The LTX2 trainer already solves the corresponding operator problem by saving the original checkpoint and then running an architecture-specific `post_save_checkpoint_hook()` that writes `<checkpoint>.comfy.safetensors`. The shared trainer already owns the CLI flags and cleanup behavior: `--no_convert_to_comfy`, `--save_original_lora`, and `--no_save_original_lora`.

## Goals / Non-Goals

**Goals:**

- Produce Krea2 `.comfy.safetensors` LoRA checkpoints by default after each saved training checkpoint.
- Preserve the original Musubi checkpoint by default for resume and internal compatibility.
- Provide a standalone Krea2 converter so existing saved Krea2 LoRAs can be converted after the fact.
- Preserve metadata and align upload/removal behavior with LTX2 and Z-Image.
- Add CPU tests for key conversion, alpha handling, post-save hook behavior, and original removal behavior.

**Non-Goals:**

- Do not change Krea2 training targets or LoRA creation.
- Do not change sampling-LoRA normalization from Comfy/native keys into Musubi keys.
- Do not require a GPU or full Krea2 model weights for converter tests.
- Do not make `.comfy.safetensors` usable as a resume checkpoint; resume remains original-format only.

## Decisions

- Implement a Krea2-specific converter rather than using generic `convert_lora.convert_to_diffusers()`. Generic underscore-to-dot conversion corrupts Krea2 module names such as `txtfusion.layerwise_blocks` and `txtfusion.refiner_blocks`.
- Use native Krea2 Comfy keys that match the Turbo LoRA reference: `diffusion_model.<module>.lora_down.weight` and `diffusion_model.<module>.lora_up.weight`. This avoids an unnecessary `lora_A` / `lora_B` dialect mismatch for Krea2.
- Drop `.alpha` keys from the Comfy output when `alpha == rank`. If `alpha != rank`, fold the alpha scaling into the up weight before dropping alpha so Comfy's alpha-less layout preserves the effective adapter strength.
- Add `Krea2NetworkTrainer.post_save_checkpoint_hook()` using the LTX2/Z-Image pattern. The hook should respect `convert_to_comfy`, upload the Comfy file when HuggingFace upload is configured, and remove the original when `save_original_lora` is false.
- Keep checkpoint rotation in the shared trainer unchanged. It already removes matching `.comfy.safetensors` files when `convert_to_comfy` is enabled.

## Risks / Trade-offs

- Incorrect name restoration for underscore-bearing modules -> Use explicit Krea2 path conversion tests for `layerwise_blocks`, `refiner_blocks`, `txtmlp`, `tmlp`, `tproj`, `last.linear`, and main blocks.
- Alpha scaling mismatch -> Test both `alpha == rank` and `alpha != rank`; fold non-default alpha into the up weight.
- Comfy loader dialect uncertainty -> Match the local Krea2 Turbo LoRA key style instead of LTX/Z-Image `lora_A` / `lora_B`.
- Operators may try to resume from `.comfy.safetensors` -> Document that resume uses the original non-Comfy checkpoint.
