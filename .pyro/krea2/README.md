# Krea2 Training Helper

`.pyro/krea2/train.sh` is a small environment-variable driven wrapper for Krea2
LoRA training. It keeps dataset selection, output naming, cache refreshes,
sampling LoRA, fp8 base loading, and LoRA target presets in one place.

Run it from the repo root:

```bash
NAME=msplits .pyro/krea2/train.sh
```

## Dataset And Output Names

`NAME` selects the dataset and prompt config pair:

```text
.pyro/krea2/cfg/${NAME}.toml
.pyro/krea2/cfg/p_${NAME}.toml
```

The actual run name is always:

```text
${NAME}-${LORA_CONFIG}
```

So this:

```bash
NAME=msplits LORA_CONFIG=preset-1 .pyro/krea2/train.sh
```

writes to:

```text
/home/pyro/models/_out/krea2/msplits-preset-1
```

and uses:

```text
--output_name msplits-preset-1
```

This makes preset experiments hard to overwrite accidentally.

## Cache Control

By default, the script skips caching because most repeat runs reuse existing
cache files:

```bash
NAME=msplits .pyro/krea2/train.sh
```

Refresh latents and text encoder outputs before training:

```bash
NAME=msplits CACHE_DATASET=1 .pyro/krea2/train.sh
```

Accepted truthy values are `1`, `true`, `yes`, and `on`. Accepted falsy values
are `0`, `false`, `no`, and `off`.

Optional cache batch-size overrides:

```bash
CACHE_DATASET=1 CACHE_LATENTS_BATCH_SIZE=2 CACHE_TEXT_BATCH_SIZE=1 .pyro/krea2/train.sh
```

## LoRA Target Presets

Set `LORA_CONFIG` to choose which Krea2 Linear layers receive LoRA modules.

```text
default   all 264 Krea2 DiT Linear layers
preset-1  main blocks attention only: wq, wk, wv, wo, gate
preset-2  main blocks attention plus MLP: gate, up, down
preset-3  diffusion-pipe-ish: any Linear module with blocks. in its path
custom    use raw NETWORK_ARGS
```

Examples:

```bash
NAME=msplits LORA_CONFIG=default .pyro/krea2/train.sh
NAME=msplits LORA_CONFIG=preset-1 .pyro/krea2/train.sh
NAME=msplits LORA_CONFIG=preset-2 .pyro/krea2/train.sh
NAME=msplits LORA_CONFIG=preset-3 .pyro/krea2/train.sh
```

For a fully custom regex filter, use `LORA_CONFIG=custom` with raw Musubi
`--network_args` entries:

```bash
NAME=msplits LORA_CONFIG=custom \
NETWORK_ARGS="include_patterns=['blocks\\.[0-9]+\\.attn\\.(wq|wk|wv|wo|gate)']" \
.pyro/krea2/train.sh
```

You can also append extra network args to a preset:

```bash
NAME=msplits LORA_CONFIG=preset-1 NETWORK_ARGS="verbose=True" .pyro/krea2/train.sh
```

## Krea2 Projector Bypass

Some Krea2 filter-bypass files are direct `txtfusion.projector` diffs, not
normal LoRA rank-pair checkpoints. Use `BYPASS` for those files instead of
`--base_weights`, `--network_weights`, or sampling LoRA args.

```bash
NAME=msplits \
BYPASS=/home/pyro/models/comfy/loras/krea/krea2filterbypass3.safetensors \
BYPASS_WEIGHT=5 \
.pyro/krea2/train.sh
```

`BYPASS` is disabled by default. When enabled, training and in-training
snapshots use the same patched Krea2 transformer. LoRAs trained with a bypass
should normally be sampled and inferred with the same bypass file and weight:

```bash
python src/musubi_tuner/krea2_generate_image.py \
  "a woman doing a cheststand pose" \
  --dit "$DIT" \
  --vae "$VAE" \
  --text_encoder "$TENC" \
  --save_path /tmp/krea2 \
  --bypass /home/pyro/models/comfy/loras/krea/krea2filterbypass3.safetensors \
  --bypass-weight 5
```

To also write a single ComfyUI companion file that embeds the scaled bypass diff
next to every saved LoRA checkpoint, enable `BYPASS_MERGE`:

```bash
NAME=msplits \
BYPASS=/home/pyro/models/comfy/loras/krea/krea2filterbypass3.safetensors \
BYPASS_WEIGHT=5 \
BYPASS_MERGE=1 \
.pyro/krea2/train.sh
```

For a checkpoint named:

```text
msplits-default-adamw8bit-step00000100.safetensors
```

the regular Comfy export remains:

```text
msplits-default-adamw8bit-step00000100.comfy.safetensors
```

and the merged export is:

```text
msplits-default-adamw8bit-step00000100.comfy.bypassed.w5.safetensors
```

Load the `.comfy.bypassed.w5.safetensors` file at strength `1.0` to reproduce
the training-time bypass weight. In loaders that treat every tensor in the file
as one adapter, lowering or raising the global LoRA strength can also scale the
embedded bypass diff.

## Training Knobs

Common overrides:

```text
MAX_STEPS       default 2000
SAVE_EVERY      default 100
SAMPLE_EVERY    default 50
GRAD_ACCUM      default 1
NETWORK_DIM     default 16
NETWORK_ALPHA   default 16
OPTIMIZER       adamw8bit | adafactor | prodigy
```

Examples:

```bash
NAME=msplits MAX_STEPS=1000 SAVE_EVERY=100 SAMPLE_EVERY=25 .pyro/krea2/train.sh
NAME=msplits NETWORK_DIM=32 NETWORK_ALPHA=32 LORA_CONFIG=preset-1 .pyro/krea2/train.sh
NAME=msplits OPTIMIZER=prodigy .pyro/krea2/train.sh
```

## Block Swap And Sampling Speed

Training block swap defaults to `BLOCKS_TO_SWAP=10`.

Snapshot VAE decode offload defaults to `SAMPLE_WITH_OFFLOADING=1`. This passes
`--sample_with_offloading`, so Krea2 moves the DiT to CPU after denoising and
before VAE decode. It lowers snapshot decode VRAM at the cost of extra transfer
latency.

Snapshot sampling has its own override:

```text
SAMPLE_WITH_OFFLOADING=1       offload DiT before snapshot VAE decode
SAMPLE_WITH_OFFLOADING=0       keep DiT resident during snapshot VAE decode
SAMPLE_BLOCKS_TO_SWAP=0        disable block swap while sampling, faster snapshots if VRAM allows
SAMPLE_BLOCKS_TO_SWAP=inherit  use the training block-swap setting
SAMPLE_BLOCKS_TO_SWAP=N        use N swapped blocks only during sampling
```

Examples:

```bash
NAME=msplits BLOCKS_TO_SWAP=10 SAMPLE_BLOCKS_TO_SWAP=0 .pyro/krea2/train.sh
NAME=msplits BLOCKS_TO_SWAP=10 SAMPLE_BLOCKS_TO_SWAP=inherit .pyro/krea2/train.sh
NAME=msplits SAMPLE_WITH_OFFLOADING=0 .pyro/krea2/train.sh
```

## Model Paths

The wrapper defaults to Pyro's local Comfy model layout:

```text
DIT        /home/pyro/models/comfy/diffusion_models/krea2_raw_bf16.safetensors
TENC       /home/pyro/models/comfy/text_encoders/qwen3vl_4b_bf16.safetensors
VAE        /home/pyro/models/comfy/vae/qwen_image_vae.safetensors
TURBO_LORA /home/pyro/models/comfy/loras/krea/krea2_turbo_lora_rank_64_bf16.safetensors
```

Override them when running elsewhere:

```bash
DIT=/workspace/models/krea2_raw_bf16.safetensors \
TENC=/workspace/models/qwen3vl_4b_bf16.safetensors \
VAE=/workspace/models/qwen_image_vae.safetensors \
TURBO_LORA=/workspace/models/krea2_turbo_lora_rank_64_bf16.safetensors \
NAME=msplits .pyro/krea2/train.sh
```

## Typical Flow

Cache once:

```bash
NAME=msplits CACHE_DATASET=1 LORA_CONFIG=default .pyro/krea2/train.sh
```

Then compare presets without re-caching:

```bash
NAME=msplits LORA_CONFIG=default .pyro/krea2/train.sh
NAME=msplits LORA_CONFIG=preset-1 .pyro/krea2/train.sh
NAME=msplits LORA_CONFIG=preset-2 .pyro/krea2/train.sh
NAME=msplits LORA_CONFIG=preset-3 .pyro/krea2/train.sh
```
