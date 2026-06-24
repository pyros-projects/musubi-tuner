## ADDED Requirements

### Requirement: Krea 2 architecture is registered in the LTX dataset layout
The system SHALL register Krea 2 as a supported image architecture in the LTX fork's existing dataset/cache layout without introducing upstream's split `dataset/architectures.py` or `dataset/cache_io.py` modules.

#### Scenario: Blueprint generation uses Krea 2 architecture
- **WHEN** a Krea2 cache or training script generates a dataset blueprint
- **THEN** it SHALL pass the Krea 2 architecture short name to the existing LTX blueprint generator

#### Scenario: Krea 2 bucket resolution is valid
- **WHEN** the bucket selector is created for the Krea 2 architecture
- **THEN** it SHALL use 16-pixel resolution steps compatible with Qwen-Image VAE scale 8 and Krea2 patch size 2

### Requirement: Krea 2 latent caching reuses Qwen-Image VAE behavior
The system SHALL provide a Krea2 latent cache script that encodes image datasets with the Qwen-Image VAE and saves Krea2-tagged latent caches.

#### Scenario: Latent cache script stores Krea2 latents
- **WHEN** `krea2_cache_latents.py` runs on an image dataset with a Qwen-Image VAE checkpoint
- **THEN** it SHALL save normalized single-image latents with Krea 2 cache metadata

#### Scenario: Control and edit inputs are not accepted
- **WHEN** Krea2 latent caching is invoked
- **THEN** it SHALL treat Krea 2 as plain text-to-image and SHALL NOT require or cache control/edit image latents

### Requirement: Krea 2 text encoder caching stores valid Qwen3-VL hidden states
The system SHALL provide a Krea2 text encoder cache script that loads Qwen3-VL, selects the Krea2 hidden-state layers, compacts valid text tokens, and stores varlen cache tensors.

#### Scenario: Text encoder cache stores selected valid tokens
- **WHEN** `krea2_cache_text_encoder_outputs.py` processes captions
- **THEN** it SHALL store a `varlen_krea2_vl_embed` tensor for each item containing only valid tokens with shape `(valid_len, selected_layers, hidden_size)`

#### Scenario: Qwen3-VL dependency is available
- **WHEN** the Krea2 text encoder loader imports Transformers classes
- **THEN** `Qwen3VLConfig` and `Qwen3VLForConditionalGeneration` SHALL be available from the configured dependency set

### Requirement: Krea 2 LoRA training runs against the RAW/base transformer
The system SHALL provide a Krea2 network trainer that loads the RAW/base Krea2 DiT, trains LoRA modules via `networks.lora_krea2`, and computes Krea2 flow-matching targets from cached latents and text embeddings.

#### Scenario: Trainer loads RAW/base DiT
- **WHEN** `krea2_train_network.py` is invoked with `--dit <raw-model>` and Krea2 caches
- **THEN** it SHALL load the Krea2 base transformer and SHALL NOT require the Turbo DiT checkpoint for training

#### Scenario: Default Krea2 LoRA targets all Linear modules
- **WHEN** `--network_module networks.lora_krea2` is used without include/exclude overrides
- **THEN** the created LoRA network SHALL target all Linear modules in the Krea2 DiT that upstream Musubi targets by default

#### Scenario: Training forward uses Krea2 cached embeddings
- **WHEN** a training batch contains `krea2_vl_embed` varlen tensors and image latents
- **THEN** the trainer SHALL pad and pack text/image tokens into the Krea2 DiT format and compute the flow target `noise - latents`

### Requirement: Krea 2 scaled fp8 is supported without plain fp8 casting
The system SHALL support Krea2 scaled-fp8 transformer loading by quantizing only safe repeated block Linear weights and preserving sensitive weights in bf16.

#### Scenario: Scaled fp8 is accepted
- **WHEN** Krea2 training or inference is invoked with `--fp8_base --fp8_scaled`
- **THEN** the loader SHALL quantize Krea2 main block Linear weights under `blocks.` while excluding `mod.`, `norm`, and `txtfusion`

#### Scenario: Plain fp8 is rejected
- **WHEN** Krea2 training or inference is invoked with `--fp8_base` but without `--fp8_scaled`
- **THEN** the command SHALL fail with a clear error explaining that Krea2 only supports scaled fp8

#### Scenario: Sampling LoRA can apply on fp8 weights
- **WHEN** Krea2 snapshot sampling applies a sampling LoRA to fp8-stored Linear modules
- **THEN** the existing sampling-LoRA runtime overlay path SHALL be used where direct merge into float8 weights is unsafe

### Requirement: Krea 2 snapshot sampling supports RAW and Turbo-LoRA preview
The system SHALL generate Krea2 snapshot samples during training using cached sample prompt embeddings and SHALL support temporary Turbo LoRA application through the existing sampling-LoRA adapter path.

#### Scenario: Sample prompts are encoded and text encoder is released
- **WHEN** Krea2 training prepares sample prompts
- **THEN** it SHALL encode prompts with Qwen3-VL, store the small embeddings on CPU, delete or offload the text encoder, and clear CUDA memory before training proceeds

#### Scenario: Global Turbo LoRA sampling adapter is applied only during snapshots
- **WHEN** Krea2 training is configured with `--sampling_lora_weight <turbo-lora>` and snapshots run
- **THEN** the trainer SHALL apply the sampling LoRA before snapshot denoising and restore the RAW/base training state afterward

#### Scenario: Prompt-local sampling LoRAs remain compatible
- **WHEN** sample prompts contain `resolved_loras`
- **THEN** Krea2 snapshot sampling SHALL use the existing prompt-local LoRA behavior and SHALL NOT also apply the global sampling LoRA for that sampling run

#### Scenario: Turbo preview schedule can use fixed mu
- **WHEN** a Krea2 snapshot sample requests `mu=1.15`
- **THEN** the Krea2 sampler SHALL use fixed `mu=1.15` for its timestep schedule instead of the RAW resolution-aware interpolation

### Requirement: Krea 2 sampling LoRA keys are normalized for local Turbo adapters
The system SHALL normalize local Krea2 Turbo LoRA weights that use native `diffusion_model.*` keys into the Musubi `lora_unet_*` key layout before applying them as sampling adapters.

#### Scenario: Diffusion model keys are converted
- **WHEN** a sampling LoRA contains keys such as `diffusion_model.blocks.0.attn.wq.lora_down.weight`
- **THEN** the Krea2 trainer SHALL convert them to the matching `lora_unet_blocks_0_attn_wq.lora_down.weight` keys before creating the sampling LoRA network

#### Scenario: Existing Musubi LoRA keys pass through
- **WHEN** a sampling LoRA already uses `lora_unet_*` keys
- **THEN** Krea2 normalization SHALL leave those keys usable without requiring native-key conversion

### Requirement: Krea 2 timestep sampling includes resolution-aware training mode
The system SHALL add `krea2_shift` as a supported timestep sampling mode in the LTX trainer.

#### Scenario: Parser accepts Krea2 shift
- **WHEN** a training command includes `--timestep_sampling krea2_shift`
- **THEN** the parser SHALL accept the value

#### Scenario: Krea2 shift uses Krea2 image sequence endpoints
- **WHEN** the trainer samples timesteps with `krea2_shift`
- **THEN** it SHALL use Krea2's sequence-length interpolation endpoints corresponding to 256px and 1280px images

### Requirement: Shared attention supports Krea 2 grouped-query attention
The system SHALL support Krea2 grouped-query attention across the shared attention backends used by LTX Musubi.

#### Scenario: SDPA receives expanded key/value heads
- **WHEN** Krea2 attention has more query heads than key/value heads and uses the torch attention path
- **THEN** key/value heads SHALL be repeated to match query heads before SDPA is called

#### Scenario: Non-contiguous attention tensors do not fail varlen backends
- **WHEN** Krea2 passes non-contiguous attention tensors to flash or sage varlen paths
- **THEN** the shared attention code SHALL reshape tensors safely instead of requiring contiguous `view()` semantics

### Requirement: Krea 2 standalone inference is available for smoke testing
The system SHALL provide a Krea2 image generation script that loads Krea2 DiT, Qwen-Image VAE, Qwen3-VL text encoder, optional LoRA weights, scaled fp8, and block swap.

#### Scenario: Standalone generation runs from one prompt
- **WHEN** `krea2_generate_image.py` is invoked with model paths and a prompt
- **THEN** it SHALL encode the prompt, denoise with Krea2, decode with Qwen-Image VAE, and save an image

#### Scenario: LoRA inference merges at load time under fp8
- **WHEN** standalone Krea2 generation uses `--fp8_scaled` and `--lora_weight`
- **THEN** the LoRA SHALL be merged into the base state before fp8 quantization is applied

### Requirement: Krea 2 documentation and CPU tests are provided
The system SHALL document the Krea2 LTX workflow and provide CPU-testable coverage for key pure-Python behavior.

#### Scenario: Documentation describes supported preview path
- **WHEN** a user reads the Krea2 documentation
- **THEN** it SHALL describe RAW/base training plus Turbo LoRA sampling adapter preview and SHALL mark full Turbo-DiT swapping as out of initial scope

#### Scenario: CPU tests validate pure behavior
- **WHEN** tests are run without a GPU or model weights
- **THEN** they SHALL cover Krea2 text compaction, `krea2_shift`, and sampling-LoRA key normalization
