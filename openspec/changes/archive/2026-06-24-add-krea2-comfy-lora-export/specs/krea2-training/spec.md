## ADDED Requirements

### Requirement: Krea 2 LoRA checkpoints export ComfyUI companion files
The system SHALL convert saved Krea2 LoRA checkpoints from Musubi `lora_unet_*` format to native Krea2 ComfyUI format by default. The converted file SHALL be saved next to the original checkpoint with `.comfy.safetensors` inserted before the extension. The original Musubi checkpoint SHALL remain available by default for resume and internal workflows.

#### Scenario: Saved Krea2 checkpoint writes a Comfy companion
- **WHEN** `krea2_train_network.py` saves `name-step00000100.safetensors` with default save flags
- **THEN** it SHALL also save `name-step00000100.comfy.safetensors`
- **AND** the original `name-step00000100.safetensors` SHALL remain present

#### Scenario: Krea2 Comfy conversion preserves native Krea2 module paths
- **WHEN** a Krea2 LoRA checkpoint contains keys such as `lora_unet_blocks_0_attn_wq.lora_down.weight`, `lora_unet_txtfusion_layerwise_blocks_0_attn_wq.lora_up.weight`, and `lora_unet_txtfusion_refiner_blocks_1_mlp_down.lora_down.weight`
- **THEN** the converted checkpoint SHALL contain native keys such as `diffusion_model.blocks.0.attn.wq.lora_down.weight`, `diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lora_up.weight`, and `diffusion_model.txtfusion.refiner_blocks.1.mlp.down.lora_down.weight`
- **AND** it SHALL NOT produce corrupted paths such as `txtfusion.layerwise.blocks` or `txtfusion.refiner.blocks`

#### Scenario: Krea2 Comfy conversion handles alpha values
- **WHEN** a Krea2 LoRA module has an `.alpha` value equal to its rank
- **THEN** the converted Comfy checkpoint SHALL omit the `.alpha` key without changing the effective up/down weights
- **WHEN** a Krea2 LoRA module has an `.alpha` value different from its rank
- **THEN** the converted Comfy checkpoint SHALL fold the alpha/rank scale into the up weight before omitting the `.alpha` key

#### Scenario: Krea2 Comfy conversion can be disabled
- **WHEN** Krea2 training is invoked with `--no_convert_to_comfy`
- **THEN** saved Krea2 LoRA checkpoints SHALL remain in original Musubi format only

#### Scenario: Krea2 original checkpoint can be removed after conversion
- **WHEN** Krea2 training is invoked with `--no_save_original_lora` and conversion succeeds
- **THEN** the converted `.comfy.safetensors` checkpoint SHALL remain and the original non-Comfy checkpoint SHALL be removed

#### Scenario: Krea2 Comfy checkpoint preserves metadata
- **WHEN** a Krea2 LoRA checkpoint contains safetensors metadata
- **THEN** the converted `.comfy.safetensors` checkpoint SHALL preserve that metadata

#### Scenario: Standalone Krea2 converter converts existing checkpoints
- **WHEN** an operator runs the Krea2 LoRA-to-Comfy converter on an existing Krea2 `lora_unet_*` checkpoint
- **THEN** it SHALL write the same native Krea2 Comfy key layout used by automatic post-save export

#### Scenario: Krea2 documentation describes export formats
- **WHEN** a user reads the Krea2 documentation
- **THEN** it SHALL describe the default dual-save behavior, the `--no_convert_to_comfy` and `--no_save_original_lora` flags, and the requirement to resume training from the original non-Comfy checkpoint
