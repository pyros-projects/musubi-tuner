# boogu-image-training Specification

## Purpose
Boogu Image Base text-to-image LoRA training support, including Boogu-specific caching, training, snapshot sampling, fp8-safe loading, and checkpoint conversion.

## Requirements
### Requirement: Boogu Image architecture registration
The system SHALL register Boogu Image Base as a distinct image-training architecture named `boogu_image`.

#### Scenario: Dataset blueprint selects Boogu Image
- **WHEN** a Boogu cache or training script creates a dataset blueprint
- **THEN** the dataset architecture SHALL be recorded as `boogu_image`

#### Scenario: Resolution buckets align to Boogu patching
- **WHEN** Boogu Image datasets are bucketed for latent caching or training
- **THEN** width and height SHALL align to the model's VAE scale and transformer patch requirements

### Requirement: Boogu latent caching
The system SHALL provide a Boogu Image latent cache script that encodes images with the Boogu/FLUX-compatible AutoencoderKL VAE and saves Boogu-tagged latent caches.

#### Scenario: Latent cache stores Boogu latents
- **WHEN** `boogu_image_cache_latents.py` runs on an image dataset with a Boogu-compatible VAE checkpoint
- **THEN** each cached item SHALL contain a latent tensor with Boogu Image architecture metadata

#### Scenario: Latent cache uses normalized VAE convention
- **WHEN** Boogu Image latents are saved
- **THEN** the cache SHALL apply the VAE `scaling_factor` and `shift_factor` convention expected by Boogu sampling and training

### Requirement: Boogu text encoder caching
The system SHALL provide a Boogu Image text encoder cache script that encodes captions with Qwen3-VL instruction features and stores variable-length per-caption tensors.

#### Scenario: Text cache stores instruction features
- **WHEN** `boogu_image_cache_text_encoder_outputs.py` processes captions
- **THEN** each cached item SHALL contain a Boogu instruction feature tensor for the caption

#### Scenario: Text cache preserves variable length
- **WHEN** captions have different token lengths
- **THEN** cached instruction features SHALL keep their natural token length rather than requiring global padding

#### Scenario: Empty or prefix-only captions remain trainable
- **WHEN** a dataset item has no sidecar caption but a dataset or global caption prefix supplies text
- **THEN** the Boogu text cache SHALL encode that resulting caption using the same path as sidecar captions

### Requirement: Boogu LoRA training
The system SHALL provide a Boogu Image network trainer that trains LoRA adapters on the Boogu Base transformer using cached latents and cached instruction features.

#### Scenario: Trainer loads Boogu Base components
- **WHEN** `boogu_image_train_network.py` starts with Boogu transformer, VAE, and text-embedding cache inputs
- **THEN** it SHALL load the Boogu Base transformer and VAE, avoid loading Qwen3-VL during the train loop, and prepare the model for LoRA training

#### Scenario: Trainer computes Boogu loss target
- **WHEN** a training batch is processed
- **THEN** the trainer SHALL convert Musubi timesteps to Boogu native flow time and compare the model prediction against the correct velocity target

#### Scenario: Trainer supports gradient checkpointing
- **WHEN** `--gradient_checkpointing` is enabled
- **THEN** Boogu transformer blocks SHALL use checkpointing without changing tensor shapes or output semantics

### Requirement: Boogu LoRA target module
The system SHALL provide a Boogu-specific LoRA network module that wraps trainable transformer Linear modules and skips unsafe or unused projections.

#### Scenario: LoRA module targets Boogu transformer layers
- **WHEN** `--network_module networks.lora_boogu_image` is used
- **THEN** LoRA adapters SHALL be created for the intended Boogu transformer attention/feed-forward/projection layers

#### Scenario: LoRA module avoids deleted projections
- **WHEN** the Boogu double-stream attention wrapper has deleted unused q/k/v projections
- **THEN** LoRA creation SHALL NOT expect those deleted projections to exist

### Requirement: Boogu snapshot sampling
The system SHALL generate Boogu Image snapshot samples during training using cached sample prompt embeddings and the native Boogu sampling schedule.

#### Scenario: Sample prompts are cached before training
- **WHEN** sample prompts are configured
- **THEN** Qwen3-VL SHALL encode those prompts before training and the text encoder SHALL be released before the train loop resumes

#### Scenario: Snapshot sampler uses Boogu native time
- **WHEN** snapshot generation runs
- **THEN** the sampler SHALL use Boogu native flow time, resolution-aware time shift, and the configured step count

#### Scenario: Snapshot sampler supports CFG
- **WHEN** a negative prompt or CFG scale is configured for a Boogu sample
- **THEN** the sampler SHALL generate conditional and unconditional predictions and combine them with the configured guidance scale

### Requirement: Boogu fp8-safe loading
The system SHALL support Boogu training from bf16 safetensors and optionally apply a safe fp8/scaled-fp8 transform without relying on the HF torchao `.bin` fp8 release.

#### Scenario: bf16 checkpoint loads by default
- **WHEN** a Boogu Base bf16 transformer checkpoint is configured
- **THEN** the trainer SHALL load it without requiring torchao `.bin` deserialization

#### Scenario: direct torchao fp8 release is rejected or deferred
- **WHEN** a user config points directly at a Boogu `-fp8` torchao `.bin` release path unsupported by the trainer
- **THEN** the trainer SHALL fail with a clear error or documented unsupported-state message

#### Scenario: safe fp8 keeps critical modules in higher precision
- **WHEN** optional fp8 loading is enabled
- **THEN** norms, embeddings, modulation, and other critical non-target modules SHALL remain in a safe dtype

### Requirement: Boogu checkpoint conversion
The system SHALL save Boogu LoRA checkpoints in the normal Musubi format and optionally emit a native/Comfy-compatible checkpoint.

#### Scenario: Post-save hook writes converted checkpoint
- **WHEN** a Boogu LoRA checkpoint is saved and conversion is enabled
- **THEN** the trainer SHALL write a sibling converted checkpoint with native Boogu `diffusion_model.*` keys

#### Scenario: Conversion preserves alpha scaling
- **WHEN** Musubi LoRA tensors include non-default alpha values
- **THEN** conversion SHALL fold alpha scaling into the converted tensor weights or otherwise preserve equivalent inference behavior

#### Scenario: Conversion can keep or remove original checkpoint
- **WHEN** the save-original flag is enabled or disabled
- **THEN** the post-save hook SHALL respectively preserve or remove the original Musubi checkpoint after conversion

### Requirement: Boogu examples and docs
The system SHALL document Boogu Image Base training and provide local example scripts/configs analogous to the Krea2 workflow.

#### Scenario: Documentation explains supported scope
- **WHEN** a user reads the Boogu documentation
- **THEN** it SHALL identify supported Base T2I LoRA training and call out Edit/Turbo/direct-fp8 limitations

#### Scenario: Example config supports local training
- **WHEN** a user opens `.pyro/boogu` examples
- **THEN** they SHALL find commands/configs for cache creation, a minimal LoRA training run, snapshot sampling, and optional fp8 mode
