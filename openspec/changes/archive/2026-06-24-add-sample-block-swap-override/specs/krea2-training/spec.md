## ADDED Requirements

### Requirement: Krea 2 snapshot sampling can override block swap
The system SHALL allow Krea2 in-training snapshot sampling to temporarily override the train-time block-swap count and restore the original training block-swap state afterward.

#### Scenario: Sampling inherits training block swap by default
- **WHEN** Krea2 training uses `--blocks_to_swap` and `--sample_blocks_to_swap` is not specified
- **THEN** snapshot sampling SHALL use the same block-swap count as training

#### Scenario: Sampling can disable block swap temporarily
- **WHEN** Krea2 training uses `--blocks_to_swap N` and `--sample_blocks_to_swap 0`
- **THEN** snapshot sampling SHALL move all Krea2 main block weights to the accelerator and SHALL run denoising without block swap

#### Scenario: Sampling can use an explicit block-swap count
- **WHEN** Krea2 training uses `--blocks_to_swap N` and `--sample_blocks_to_swap M` with a valid non-negative integer
- **THEN** snapshot sampling SHALL use `M` swapped Krea2 main blocks during denoising instead of `N`

#### Scenario: Training block swap is restored after successful sampling
- **WHEN** snapshot sampling completes after applying a sampling block-swap override
- **THEN** the trainer SHALL restore the original train-time block-swap count and prepare the transformer for training before the next training step

#### Scenario: Training block swap is restored after sampling failure
- **WHEN** snapshot sampling raises an exception after applying a sampling block-swap override
- **THEN** the trainer SHALL restore the original train-time block-swap count before propagating the failure

#### Scenario: Invalid sampling block-swap requests fail clearly
- **WHEN** `--sample_blocks_to_swap` is outside the valid Krea2 block-swap range or requires an offloader that was not initialized
- **THEN** the command SHALL fail with a clear error explaining why the sampling override cannot be applied
