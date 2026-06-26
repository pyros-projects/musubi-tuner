## MODIFIED Requirements

### Requirement: Boogu snapshot sampling
The system SHALL generate Boogu Image snapshot samples during training using cached sample prompt embeddings and the native Boogu sampling schedule, with visible denoise progress and memory-safe VAE decode when sample offloading is enabled.

#### Scenario: Sample prompts are cached before training
- **WHEN** sample prompts are configured
- **THEN** Qwen3-VL SHALL encode those prompts before training and the text encoder SHALL be released before the train loop resumes

#### Scenario: Snapshot sampler uses Boogu native time
- **WHEN** snapshot generation runs
- **THEN** the sampler SHALL use Boogu native flow time, resolution-aware time shift, and the configured step count

#### Scenario: Snapshot sampler supports CFG
- **WHEN** a negative prompt or CFG scale is configured for a Boogu sample
- **THEN** the sampler SHALL generate conditional and unconditional predictions and combine them with the configured guidance scale

#### Scenario: Snapshot sampler reports denoise progress
- **WHEN** Boogu snapshot generation enters the native denoise loop
- **THEN** the sampler SHALL expose a progress indicator whose total matches the number of Boogu timestep transitions for that sample

#### Scenario: Decode offload avoids transformer and VAE memory stacking
- **WHEN** Boogu snapshot sampling runs with `--sample_with_offloading`
- **THEN** the transformer SHALL be moved off the sampling device before the VAE is moved onto that device for latent decode
- **AND** pending block-swap transfers SHALL be synchronized before device cleanup when the transformer exposes such synchronization

#### Scenario: Decode offload remains optional
- **WHEN** Boogu snapshot sampling runs without `--sample_with_offloading`
- **THEN** VAE decode SHALL preserve the existing device movement behavior and SHALL NOT add an extra transformer offload before decode

#### Scenario: Decode cleanup runs on failure
- **WHEN** Boogu VAE decode raises an exception during snapshot sampling
- **THEN** the VAE SHALL still be moved back to CPU before the exception propagates
