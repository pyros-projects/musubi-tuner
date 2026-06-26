## ADDED Requirements

### Requirement: Krea 2 Comfy export can include a scaled bypass patch
The system SHALL allow Krea2 training to optionally emit an additional ComfyUI checkpoint that contains both the trained LoRA tensors and the scaled projector bypass diff used during training.

#### Scenario: Bypass-merged companion is written on save
- **WHEN** `krea2_train_network.py` saves `name-step00000100.safetensors` after being invoked with `--bypass <path> --bypass-weight 5 --bypass-merge`
- **THEN** it SHALL write the normal `name-step00000100.comfy.safetensors` companion
- **AND** it SHALL also write `name-step00000100.comfy.bypassed.w5.safetensors`
- **AND** the original `name-step00000100.safetensors` SHALL follow the existing `--save_original_lora` behavior

#### Scenario: Bypass tensor is scaled into merged output
- **WHEN** the bypass source contains a supported projector diff tensor `D`
- **AND** `--bypass-weight 5 --bypass-merge` is active
- **THEN** the bypass-merged Comfy checkpoint SHALL contain `diffusion_model.txtfusion.projector.diff` with value `D * 5`
- **AND** loading the bypass-merged Comfy checkpoint at strength `1.0` SHALL represent the same bypass magnitude used during training, subject to the consumer's normal LoRA strength handling

#### Scenario: Normal Comfy export is unchanged
- **WHEN** `--bypass-merge` is active
- **THEN** the normal `.comfy.safetensors` file SHALL NOT include the bypass projector diff
- **AND** the bypass-merged file SHALL be the only additional artifact that includes the scaled bypass projector diff

#### Scenario: Bypass merge is disabled by default
- **WHEN** Krea2 training is invoked with `--bypass <path> --bypass-weight 5` but without `--bypass-merge`
- **THEN** it SHALL apply the bypass during training and snapshots as before
- **AND** it SHALL NOT write a `.comfy.bypassed.w5.safetensors` companion

#### Scenario: Bypass merge requires an active bypass
- **WHEN** Krea2 training is invoked with `--bypass-merge` but without `--bypass`
- **THEN** startup SHALL fail before training begins
- **AND** the error SHALL explain that bypass-merged export requires a bypass path

#### Scenario: Bypass merge requires Comfy conversion
- **WHEN** Krea2 training is invoked with both `--bypass-merge` and `--no_convert_to_comfy`
- **THEN** startup SHALL fail before training begins
- **AND** the error SHALL explain that bypass-merged export depends on Comfy conversion

#### Scenario: Bypass-merged filename encodes the weight safely
- **WHEN** a bypass-merged Comfy checkpoint is written
- **THEN** its filename SHALL include a safe weight token derived from the effective bypass weight
- **AND** common weights SHALL be encoded as `5 -> w5`, `0.5 -> w0p5`, and `-1 -> wm1`

#### Scenario: Bypass-merged checkpoint preserves and extends metadata
- **WHEN** a bypass-merged Comfy checkpoint is written from a LoRA checkpoint with metadata
- **THEN** it SHALL preserve the source checkpoint metadata
- **AND** it SHALL include metadata that records the bypass source path, effective bypass weight, and that the bypass diff was merged into the Comfy checkpoint

#### Scenario: Local training script can request bypass-merged export
- **WHEN** `.pyro/krea2/train.sh` is run with an enabled bypass-merge environment flag and a non-empty bypass path
- **THEN** it SHALL pass `--bypass-merge` to `krea2_train_network.py`

#### Scenario: Documentation warns about strength coupling
- **WHEN** a user reads the Krea2 local training documentation
- **THEN** it SHALL explain that the bypass-merged Comfy file is intended to be loaded at strength `1.0`
- **AND** it SHALL explain that changing the global LoRA strength in loaders that treat all tensors as one adapter can also scale the embedded bypass diff
