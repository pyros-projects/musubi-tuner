# krea2-training Specification

## Purpose
TBD - created by archiving change port-krea2-to-ltx-musubi. Update Purpose after archive.
## Requirements
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

### Requirement: Krea 2 sampling LoRA keys are normalized for local Turbo adapters
The system SHALL normalize local Krea2 Turbo LoRA weights that use native `diffusion_model.*` keys into the Musubi `lora_unet_*` key layout before applying them as sampling adapters.

#### Scenario: Diffusion model keys are converted
- **WHEN** a sampling LoRA contains keys such as `diffusion_model.blocks.0.attn.wq.lora_down.weight`
- **THEN** the Krea2 trainer SHALL convert them to the matching `lora_unet_blocks_0_attn_wq.lora_down.weight` keys before creating the sampling LoRA network

#### Scenario: Existing Musubi LoRA keys pass through
- **WHEN** a sampling LoRA already uses `lora_unet_*` keys
- **THEN** Krea2 normalization SHALL leave those keys usable without requiring native-key conversion

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
