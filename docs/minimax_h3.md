# MiniMax H3

## Scope

Musubi Tuner supports experimental LoRA training of the MiniMax H3 **FL2VA** checkpoint with either joint video/audio samples or image-only `T=1` samples. The separate Ref2VA checkpoint and reference-media conditioning are not supported yet. Training previews currently generate images only.

H3 is exceptionally large. BF16 compute is required, but the frozen transformer base may use either BF16 or ComfyUI INT8/ConvRot weights. Both the regular and pruned FL2VA INT8/ConvRot checkpoints are supported; NVFP4 is not. Gradient checkpointing and block swap are strongly recommended.

## Model files

Either model distribution can be used:

- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3): download the complete `FL2VA` directory. Pass the repository directory to every model argument; the scripts locate each component and all of its shards.
- [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3): download these individual files:
  - `diffusion_models/minimax_h3_fl2va_bf16.safetensors`
  - `diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors` (lower base-model memory)
  - `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` (lowest base-model memory)
  - `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`
  - `vae/minimax_h3_video_vae_fp16.safetensors`
  - `vae/minimax_h3_audio_vae_fp32.safetensors`

For example, the official files can be downloaded with:

```bash
hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/*" --local-dir models/MiniMax-H3
```

For the optimized INT8 kernels, install the optional dependency (the trainer also has a native PyTorch fallback):

```bash
uv sync --extra int8
```

## Dataset

For joint audio/video training, use a video dataset. H3 treats video as 24 fps and accepts frame counts of `17*n+5`: `5, 22, 39, 56, 73, 90, 107, 124, ...`. Other requested lengths are rounded down. `5` frames is supported by the VAE, but longer clips are normally more useful for training.

Set `source_fps` to the actual frame rate of the source videos so video resampling and audio crops stay synchronized. Source audio is resampled to stereo 32 kHz. A video with no audio track gets a matching silent audio latent.

```toml
[general]
resolution = [512, 512]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
video_directory = "/path/to/videos"
cache_directory = "/path/to/cache"
target_frames = [56, 90, 124]
frame_extraction = "head"
source_fps = 30.0
```

Spatial bucket sizes are aligned to 32 pixels (video-VAE compression 16 multiplied by the transformer's spatial patch size 2).

### Image-only LoRA

For subjects or styles that do not need temporal or audio targets, an image dataset is encoded as a one-frame video stream. `--image_audio_mode none` keeps the legacy image-only `[text | video]` sequence. `--image_audio_mode silent` instead adds duration-matched noised zero-audio rows so the packed sequence retains H3's joint audio-video geometry.

```toml
[general]
resolution = [512, 512]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
image_directory = "/path/to/images"
cache_directory = "/path/to/cache"
```

Image-only latent caching does not require `--audio_vae`. The cache stores an empty audio sentinel; `--image_audio_mode silent` expands it on the fly, so switching modes does not require recaching. Train these caches with `--audio_loss_weight 0`; a positive audio weight is rejected because there is no audio target.

## Pre-caching

Official repository layout:

```bash
python minimax_h3_cache_latents.py \
  --dataset_config path/to/dataset.toml \
  --vae models/MiniMax-H3 \
  --audio_vae models/MiniMax-H3 \
  --vae_dtype bfloat16

python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config path/to/dataset.toml \
  --text_encoder models/MiniMax-H3 \
  --tokenizer models/MiniMax-H3/FL2VA/tokenizer \
  --text_encoder_dtype bfloat16
```

For ComfyUI weights, give each corresponding `.safetensors` path instead. `--tokenizer` may point to the official `FL2VA/tokenizer` directory; if omitted, it is loaded from the official Hugging Face repository.

ComfyUI INT8/ConvRot text encoders can be cached through ComfyUI's own loader. Run this command with ComfyUI's Python environment so its runtime dependencies are available:

```bash
PYTHONPATH=src /path/to/ComfyUI/.venv/bin/python -m musubi_tuner.minimax_h3_cache_text_encoder_outputs \
  --dataset_config path/to/dataset.toml \
  --text_encoder models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --comfyui_path /path/to/ComfyUI
```

The text cache contains raw-prompt Qwen3-VL features immediately after layer 50, before the final RMSNorm. Chat templates and automatic special tokens are intentionally not applied, matching H3 FL2VA inference.

## Training

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 minimax_h3_train_network.py \
  --dit models/MiniMax-H3 \
  --dataset_config path/to/dataset.toml \
  --sdpa --mixed_precision bf16 \
  --timestep_sampling shift --discrete_flow_shift 12 \
  --weighting_scheme none --audio_loss_weight 1.0 \
  --gradient_checkpointing --blocks_to_swap 40 \
  --network_module networks.lora --network_dim 32 --lora_target_preset attn_mlp \
  --optimizer_type adamw8bit --learning_rate 1e-4 \
  --max_train_epochs 16 --save_every_n_epochs 1 \
  --output_dir path/to/output --output_name h3-lora
```

To train against a Comfy INT8/ConvRot base, point `--dit` at either FL2VA INT8 file listed above. Do not pass `--fp8_base` or `--fp8_scaled`; quantization is detected from the checkpoint metadata. The frozen base uses INT8 forward matmuls, while its input-gradient path dequantizes bounded BF16 chunks for training accuracy. The trainable LoRA and model compute remain BF16.

`--lora_target_preset attn_mlp` is the default and is portable between the full and modulation-pruned transformer layouts. The optional `full` preset also targets AdaLN projections, whose input width is `2688` in the full checkpoint but `8` in the pruned curve-table checkpoint. Use `full` only when training and inference use the same transformer layout.

`--lora_target_preset no_packed_attn` freezes attention in all 50 packed audio/video/text blocks while training their MLP projections plus every attention and MLP projection in the two text-only token-refiner blocks (108 modules). This removes direct LoRA updates from H3's shared spatiotemporal attention path; it does not make temporal behavior immutable because the trained MLP outputs still feed later frozen attention blocks.

H3 uses a video sigma shift of 12 and an audio sigma shift of 3. The trainer samples the video schedule, maps the same base time to the audio schedule, noises both cached streams, and optimizes both raw `clean-noise` velocity targets. `--audio_loss_weight` controls the audio term relative to video.

### ComfyUI-format LoRA output

Every saved checkpoint gets a `.comfy.safetensors` twin in ComfyUI/ai-toolkit key format (`diffusion_model.blocks.N....lora_A/lora_B`, kohya `alpha/rank` scaling folded into `lora_B`). Disable with `--no_convert_to_comfy`. Existing musubi-format LoRAs convert standalone:

```bash
python -m musubi_tuner.minimax_h3.convert_lora_to_comfy path/to/lora.safetensors
```

ComfyUI's regular LoRA loader accepts both formats. Note for INT8 pre-bake nodes (e.g. `INT8 Pre-Lora Loader`): LoRA weights in either format can only be baked into layers that are still high-precision at load time. Pre-quantized `*_int8_convrot` checkpoints store all packed blocks as INT8, so only the BF16 token-refiner layers bake and the rest are silently skipped — bake against the BF16/pruned-BF16 transformer with on-the-fly quantization instead.

### Image previews during training

The launcher caches all preview prompts before training and stores them in the first dataset's `cache_directory` as `minimax_h3_sample_prompts_cache.pt`. The 32B text encoder is loaded only when that cache is missing or stale, and its cache process exits before the trainer starts. Prompt files therefore contain only generation settings:

```toml
[prompt]
width = 512
height = 512
frame_count = 1
sample_steps = 12

[[prompt.subset]]
prompt = "lucy the cat"
seed = 42
```

Add the decoder and cadence to the training command:

```bash
--vae models/vae/minimax_h3_video_vae_fp16.safetensors \
--vae_dtype float16 \
--sample_prompts path/to/p_lucy.toml \
--sample_every_n_steps 100
```

The local launcher selects `.pyro/h3/cfg/p_${H3_NAME}.toml`; set `SAMPLE_PROMPTS` only to override it explicitly. Prompt changes invalidate and rebuild the aggregate cache automatically before training.

By default the trainer samples the minimum natural video packet without CFG: two temporal latents plus duration-matched silent audio rows when the training run uses them (`--sample_audio_mode auto` follows `--image_audio_mode`), integrated with a second-order multistep solver (`--sample_solver ab2`). H3 is a video model, so giving it an in-distribution packet and harvesting one frame produces markedly better stills than denoising a lone latent. The packet is reduced to one preview via `--sample_frame_select`: the default `dup_last` decodes the packet's second latent through the measured-best duplicate path, while `first`, `last`, and `sharpest` pick from the natural five-frame clip (the seam frame between the two latent patches decodes with striping and is skipped). `--sample_latent_frames 1` restores the legacy single-latent mode, whose decode duplicates the latent and keeps the final frame. Setting `frame_count` to a `17*n+5` value in a prompt turns that subset into a video preview: the full latent clip is sampled, decoded through the chunked reference decoder, and saved as a silent `.mp4` (sampled audio latents are discarded; other counts are aligned down). Before VAE decoding, the frozen transformer is temporarily parked on CPU to keep peak VRAM bounded. Preview PNGs are written under `<output_dir>/sample/`.

Notes:

- With classic block swap, keep the loader `batch_size` at `1`; use `--gradient_accumulation_steps` for a larger effective batch.
- `--blocks_to_swap` can be at most 48 for the 50-block transformer.
- `--sample_blocks_to_swap` overrides block swap during previews: leave it unset to inherit training, use `0` for faster unswapped sampling, or a smaller positive count if `0` does not fit VRAM. The local launcher defaults this to `0`; set `SAMPLE_BLOCKS_TO_SWAP=inherit` to reuse `BLOCKS_TO_SWAP`.
- `--fp8_base` and `--fp8_scaled` are rejected. INT8/ConvRot is selected by the `--dit` checkpoint itself.
- H3 training previews support single images (`frame_count=1`) and silent video clips (`frame_count` of `17*n+5`), positive prompts, and classic block swap only. Video previews are much slower per tick; prefer 512px canvases and a low sampling cadence. The preview flags `--sample_latent_frames`, `--sample_audio_mode`, `--sample_solver`, and `--sample_frame_select` control the internal sampling packet for A/B testing; each can also be set per prompt inside the prompt file (`sample_latent_frames = 1` in a `[[prompt.subset]]` overrides the CLI flag), so one run can preview several sampling configurations side by side.
- Video samples without an audio track use encoded silent audio latents. For image caches, choose `--image_audio_mode none` or `silent`; synthesized silent rows remain loss-excluded and require `--audio_loss_weight 0`.
