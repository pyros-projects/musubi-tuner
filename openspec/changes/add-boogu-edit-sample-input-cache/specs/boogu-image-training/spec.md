## MODIFIED Requirements

### Requirement: Boogu snapshot sampling
The system SHALL generate Boogu Image snapshot samples during training using cached sample prompt embeddings, optional cached edit input image latents, and the native Boogu sampling schedule.

#### Scenario: Sample prompts are cached before training
- **WHEN** sample prompts are configured
- **THEN** Qwen3-VL SHALL encode those prompts before training and the text encoder SHALL be released before the train loop resumes

#### Scenario: Snapshot sampler uses Boogu native time
- **WHEN** snapshot generation runs
- **THEN** the sampler SHALL use Boogu native flow time, resolution-aware time shift, and the configured step count

#### Scenario: Snapshot sampler supports CFG
- **WHEN** a negative prompt or CFG scale is configured for a Boogu sample
- **THEN** the sampler SHALL generate conditional and unconditional predictions and combine them with the configured guidance scale

#### Scenario: Sample prompt defines a global edit input image
- **WHEN** a Boogu sample prompt TOML defines `input_image` under `[prompt]`
- **THEN** every `[[prompt.subset]]` entry without its own edit input image SHALL use that global input image for snapshot sampling

#### Scenario: Sample prompt overrides edit input image per prompt
- **WHEN** a Boogu sample prompt TOML defines `input_image` under a `[[prompt.subset]]` entry
- **THEN** that prompt SHALL use the subset input image instead of the global input image for snapshot sampling

#### Scenario: Sample prompt accepts compatible single-image alias
- **WHEN** a Boogu sample prompt TOML defines `control_image_path` as a string or single-item list and no `input_image` is defined for that prompt
- **THEN** the sampler SHALL treat that value as the prompt's edit input image

#### Scenario: Sample prompt rejects ambiguous edit inputs
- **WHEN** a Boogu sample prompt defines multiple edit input images for one prompt
- **THEN** prompt processing SHALL fail before training with a clear error

#### Scenario: Edit input images are pre-cached before transformer loading
- **WHEN** Boogu sample prompts include edit input images
- **THEN** the trainer SHALL encode Qwen3-VL image-conditioned instruction features and VAE reference latents before loading the Boogu transformer for training

#### Scenario: Snapshot sampler uses cached edit input conditioning
- **WHEN** a Boogu sample has a cached edit input image
- **THEN** the positive denoise path SHALL use image-conditioned instruction features and both conditional denoise paths SHALL receive the cached reference image latents

#### Scenario: Text-only snapshot samples remain supported
- **WHEN** a Boogu sample prompt has no edit input image
- **THEN** the sampler SHALL keep the existing text-only prompt embedding and pass no reference image latents to the transformer
