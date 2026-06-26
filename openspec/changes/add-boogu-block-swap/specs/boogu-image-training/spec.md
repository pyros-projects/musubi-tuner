## ADDED Requirements

### Requirement: Boogu block swap support
The system SHALL support Boogu Image transformer block swap during training and snapshot sampling using the repository's shared block-offload lifecycle.

#### Scenario: Trainer accepts Boogu block swap
- **WHEN** `boogu_image_train_network.py` starts with a positive `--blocks_to_swap` value
- **THEN** the trainer SHALL enable Boogu transformer block swap instead of rejecting the configuration

#### Scenario: Block swap uses Boogu forward order
- **WHEN** Boogu block swap is enabled
- **THEN** offloadable blocks SHALL be ordered as context refiners, noise refiners, reference-image refiners, double-stream layers, and single-stream layers

#### Scenario: Forward waits and submits block transfers
- **WHEN** a Boogu forward pass runs with block swap enabled
- **THEN** the transformer SHALL wait for each managed block before executing it and SHALL submit post-block movement after executing it

#### Scenario: Device movement excludes swapped blocks
- **WHEN** the trainer moves a Boogu transformer to the training device with block swap enabled
- **THEN** non-block modules SHALL move to the target device while block modules remain excluded from that bulk move

#### Scenario: Training and inference switches update offloader mode
- **WHEN** shared training code switches Boogu block swap between training and inference modes
- **THEN** the transformer SHALL update the offloader forward-only mode and prepare block devices before the next forward pass

#### Scenario: Sampling block swap override is temporary
- **WHEN** Boogu snapshot sampling runs with `--sample_blocks_to_swap`
- **THEN** the transformer SHALL temporarily use the requested sampling block-swap count and SHALL restore the original training block-swap state after sampling succeeds or fails

#### Scenario: Invalid block swap counts fail clearly
- **WHEN** a Boogu block-swap request is negative or exceeds the supported maximum for the computed ordered block list
- **THEN** the system SHALL raise a clear error that includes the valid limit and requested value

#### Scenario: Disabled block swap preserves current behavior
- **WHEN** Boogu block swap is not enabled
- **THEN** the block-swap methods SHALL remain safe no-ops compatible with shared trainer and sampler calls
