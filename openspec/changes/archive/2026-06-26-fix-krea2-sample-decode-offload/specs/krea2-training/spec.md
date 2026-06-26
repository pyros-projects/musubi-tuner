## ADDED Requirements

### Requirement: Krea 2 sampling can offload the transformer before VAE decode
The system SHALL allow Krea2 sampling paths to move the transformer off the accelerator after denoising and before moving the Qwen-Image VAE onto the accelerator for decode when sampling offload is requested.

#### Scenario: Training snapshot offloads before VAE decode
- **WHEN** Krea2 training snapshot sampling runs with `--sample_with_offloading` on an accelerator device
- **THEN** the transformer SHALL be moved off the accelerator after denoising completes and before the VAE is moved onto the accelerator for decode
- **AND** accelerator memory SHALL be cleared at that boundary before VAE decode starts

#### Scenario: Training snapshot restores a usable sampling state after decode
- **WHEN** a Krea2 training snapshot has offloaded the transformer before VAE decode
- **THEN** the transformer SHALL be in a state that the shared sampling loop can reuse or offload safely after the sample returns
- **AND** block-swap sampling state SHALL remain restorable for the next prompt and the next training step

#### Scenario: Training snapshot restores state after decode failure
- **WHEN** Krea2 VAE decode raises after the transformer was offloaded for decode
- **THEN** the transformer SHALL be left in a state compatible with the shared sampling loop's cleanup path before the error is propagated

#### Scenario: Standalone generation can request decode offload
- **WHEN** `krea2_generate_image.py` is invoked with the Krea2 sampling offload flag
- **THEN** the standalone sampler SHALL offload the transformer after denoising and before VAE decode
- **AND** it SHALL restore the transformer before the next prompt or interactive generation step when continued sampling needs it

#### Scenario: Default sampling behavior is unchanged
- **WHEN** Krea2 sampling offload is not requested
- **THEN** Krea2 sampling SHALL keep the current resident-transformer decode behavior
