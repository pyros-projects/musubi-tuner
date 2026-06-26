## ADDED Requirements

### Requirement: Krea 2 can apply explicit projector bypass patches
The system SHALL allow Krea2 training and standalone generation commands to apply a direct `txtfusion.projector` bypass diff to the loaded Krea2 DiT before the transformer is used for training, denoising, snapshot sampling, or block-swap setup.

#### Scenario: Training applies bypass before training and snapshots
- **WHEN** `krea2_train_network.py` is invoked with `--bypass <path> --bypass-weight 5`
- **THEN** the trainer SHALL load a supported projector diff from `<path>`
- **AND** it SHALL add `diff * 5` to the loaded Krea2 DiT `txtfusion.projector.weight` before training forward passes can run
- **AND** in-training snapshot sampling SHALL use the same patched transformer without requiring a separate sampling bypass argument

#### Scenario: Standalone generation applies bypass before denoising
- **WHEN** `krea2_generate_image.py` is invoked with `--bypass <path> --bypass-weight 5`
- **THEN** standalone generation SHALL load a supported projector diff from `<path>`
- **AND** it SHALL add `diff * 5` to the loaded Krea2 DiT `txtfusion.projector.weight` before denoising any prompt

#### Scenario: Bypass does not use LoRA merge paths
- **WHEN** a bypass checkpoint contains `diffusion_model.txtfusion.projector.diff`
- **THEN** the system SHALL apply it as a projector-weight patch
- **AND** it SHALL NOT require the checkpoint to contain `lora_down`, `lora_up`, or `alpha` keys
- **AND** it SHALL NOT route the checkpoint through `--base_weights`, `--network_weights`, or `--sampling_lora_weight`

#### Scenario: Supported bypass key variants are accepted
- **WHEN** a bypass checkpoint contains one of `diffusion_model.txtfusion.projector.diff`, `txtfusion.projector.diff`, `diffusion_model.txtfusion.projector.weight.diff`, or `txtfusion.projector.weight.diff`
- **THEN** the bypass loader SHALL accept that tensor as the projector diff

#### Scenario: Omitted bypass preserves existing behavior
- **WHEN** Krea2 training or standalone generation is invoked without `--bypass`
- **THEN** the system SHALL NOT modify `txtfusion.projector.weight`
- **AND** existing LoRA, sampling LoRA, fp8, and block-swap behavior SHALL remain unchanged

#### Scenario: Invalid bypass fails clearly
- **WHEN** `--bypass` points to a checkpoint without a supported projector diff key or with a diff shape that does not match `txtfusion.projector.weight`
- **THEN** Krea2 training or standalone generation SHALL fail before training or denoising begins
- **AND** the error SHALL identify the missing key or shape mismatch

### Requirement: Krea 2 training surfaces and records bypass configuration
The system SHALL expose projector-bypass configuration through the local Krea2 training script and record active bypass settings in Krea2 LoRA checkpoint metadata.

#### Scenario: Training metadata records active bypass
- **WHEN** Krea2 training saves a LoRA checkpoint after starting with `--bypass <path> --bypass-weight 5`
- **THEN** the checkpoint metadata SHALL include the bypass path
- **AND** the checkpoint metadata SHALL include the bypass weight

#### Scenario: Training metadata is omitted when bypass is inactive
- **WHEN** Krea2 training saves a LoRA checkpoint without `--bypass`
- **THEN** the checkpoint metadata SHALL NOT claim a Krea2 bypass path or weight was active

#### Scenario: Local train script can pass bypass settings
- **WHEN** `.pyro/krea2/train.sh` is run with a non-empty bypass environment variable and a bypass weight
- **THEN** it SHALL append the corresponding `--bypass` and `--bypass-weight` arguments to the Krea2 training command

#### Scenario: Local train script leaves bypass disabled by default
- **WHEN** `.pyro/krea2/train.sh` is run without a bypass path
- **THEN** it SHALL not pass bypass arguments to the Krea2 training command
