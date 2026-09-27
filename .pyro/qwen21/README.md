# Qwen-Image 2.1 Training Helper

`.pyro/qwen21/train.sh` is the Qwen-Image 2.1 twin of `.pyro/krea2/train.sh`: the same
environment-variable interface and the same `cfg/NAME.toml` + `cfg/p_NAME.toml` pairs.
It trains LoRAs directly on the Comfy INT8 ConvRot DiT
(`qwen_image_2.1_int8_convrot.safetensors`): the quantized Linears stay frozen INT8 and run
through comfy-kitchen, only the LoRA weights train.

Run it from the repo root:

```bash
NAME=flat .pyro/qwen21/train.sh
```

## Dataset And Output Names

`NAME` selects the dataset and prompt config pair:

```text
.pyro/qwen21/cfg/${NAME}.toml
.pyro/qwen21/cfg/p_${NAME}.toml
```

The run name is `${NAME}-${LORA_CONFIG}-${OPTIMIZER}` and the run writes to:

```text
/home/pyro/models/_out/qwen21/${NAME}-${LORA_CONFIG}-${OPTIMIZER}
```

A Krea2 dataset config works unchanged apart from `cache_directory`: point it at a
`/home/pyro/models/_out/qwen21/cache/...` directory. Cache files carry the architecture in
their name (`_qi21`), so sharing a directory with Krea2 caches would also work, but separate
directories are easier to clean up.

In the prompt file, `sample_steps`, `width`, `height`, `seed`, `negative_prompt` and
`cfg_scale` mean the same as for Krea2. Differences:

- Qwen-Image 2.1 is meant to be sampled without guidance. CFG only runs when a negative
  prompt is set and `cfg_scale > 1`.
- With the turbo LoRA on (the default, see below) snapshots ignore `sample_steps` and use the
  turbo schedule. `sample_steps` only counts with `TURBO_LORA=`; the official pipeline uses 40,
  the Comfy template 25.
- Leave `mu` out. The time shift then follows the resolution, as in the official scheduler
  (0.69 at 1024x1024). Set `mu` only to pin it. Krea2's `mu = 1.15` is wrong for this model.

## Turbo Snapshots

As with Krea2, snapshots run with a turbo LoRA by default, so previews take 6 steps instead
of 25:

```text
TURBO_LORA    /home/pyro/models/comfy/loras/qwen21/turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors
TURBO_SIGMAS  1.0,0.9375,0.875,0.75,0.5,0.25
```

The turbo LoRA only touches snapshots, never the trained LoRA. It follows Viggle's contract:

- `TURBO_SIGMAS` are raw nodes. They get the resolution shift, but not the base scheduler's
  `shift_terminal` stretch, which would wreck the last step.
- CFG stays at 1, so keep `cfg_scale = 1.0` in the prompt file.
- The LoRA runs at strength 1.0, unmerged: a runtime overlay on top of the INT8 base and the
  LoRA being trained.

Its PEFT keys address the unfused `img_mlp.gate_layer`/`img_mlp.proj`; they are fused exactly
onto this DiT's `img_mlp.gate_up` when loaded.

Other Viggle schedules add steps only at the high-noise end, for example 7 steps:

```bash
NAME=flat TURBO_SIGMAS=1.0,0.9583,0.9167,0.875,0.75,0.5,0.25 .pyro/qwen21/train.sh
```

Sample on the base model instead, with the prompt file's `sample_steps`:

```bash
NAME=flat TURBO_LORA= .pyro/qwen21/train.sh
```

## Cache Control

By default the script skips caching:

```bash
NAME=flat .pyro/qwen21/train.sh
```

Refresh latents and text encoder outputs before training:

```bash
NAME=flat CACHE_DATASET=1 .pyro/qwen21/train.sh
```

Accepted truthy values are `1`, `true`, `yes`, and `on`. Accepted falsy values
are `0`, `false`, `no`, and `off`.

Optional cache batch-size overrides:

```bash
CACHE_DATASET=1 CACHE_LATENTS_BATCH_SIZE=2 CACHE_TEXT_BATCH_SIZE=1 .pyro/qwen21/train.sh
```

The text encoder is Qwen3-VL-8B (`qwen3vl_8b_int8_convrot.safetensors`, about 7.6 GB VRAM
while caching). Captions are encoded one at a time whatever the batch size, exactly as
Comfy encodes a prompt.

## LoRA Target Presets

Set `LORA_CONFIG` to choose which Qwen-Image 2.1 Linear layers receive LoRA modules.
The DiT has 32 blocks, each with `attn.to_q/to_k/to_v/to_out.0` and a fused SwiGLU MLP
`img_mlp.gate_up/out`.

```text
default   all 200 DiT Linear layers: the blocks plus img_in, txt_in, timestep MLP,
          the shared modulation, norm_out.linear and proj_out
preset-1  main blocks attention only: to_q, to_k, to_v, to_out.0 (128 layers)
preset-2  main blocks attention plus MLP: gate_up, out (192 layers)
preset-3  diffusion-pipe-ish: any Linear module with blocks. in its path (same 192 as preset-2)
custom    use raw NETWORK_ARGS
```

`default` includes the single modulation Linear that every block shares; `preset-2` is the
conventional Qwen-Image choice if that is too broad.

Examples:

```bash
NAME=flat LORA_CONFIG=default .pyro/qwen21/train.sh
NAME=flat LORA_CONFIG=preset-1 .pyro/qwen21/train.sh
NAME=flat LORA_CONFIG=preset-2 .pyro/qwen21/train.sh
```

For a fully custom regex filter, use `LORA_CONFIG=custom` with raw Musubi
`--network_args` entries:

```bash
NAME=flat LORA_CONFIG=custom \
NETWORK_ARGS="include_patterns=['transformer_blocks\\.(1[6-9]|2[0-9]|3[01])\\..*']" \
.pyro/qwen21/train.sh
```

You can also append extra network args to a preset:

```bash
NAME=flat LORA_CONFIG=preset-1 NETWORK_ARGS="verbose=True" .pyro/qwen21/train.sh
```

## ComfyUI

The DiT's module names are the Comfy checkpoint keys, so the saved
`${RUN_NAME}-step*.safetensors` LoRAs load in ComfyUI as they are, with the regular
LoRA loader on the INT8 ConvRot model (no `.comfy.safetensors` twin is needed).

## Training Knobs

Common overrides:

```text
MAX_STEPS       default 1000
SAVE_EVERY      default 100
SAMPLE_EVERY    default 50
GRAD_ACCUM      default 1
NETWORK_DIM     default 16
NETWORK_ALPHA   default 16
OPTIMIZER       adamw8bit | adafactor | prodigy
```

Examples:

```bash
NAME=flat MAX_STEPS=1000 SAVE_EVERY=100 SAMPLE_EVERY=25 .pyro/qwen21/train.sh
NAME=flat NETWORK_DIM=32 NETWORK_ALPHA=32 LORA_CONFIG=preset-1 .pyro/qwen21/train.sh
NAME=flat OPTIMIZER=prodigy .pyro/qwen21/train.sh
```

## Memory And Sampling Speed

The INT8 DiT takes about 7.3 GB, so there is no block swap and no fp8 knob:
`BLOCKS_TO_SWAP`, `SAMPLE_BLOCKS_TO_SWAP` and `BYPASS*` from the Krea2 launcher do not exist
here. The r256 turbo overlay adds about 1.8 GB (bf16) while snapshots run.

Snapshot VAE decode offload defaults to `SAMPLE_WITH_OFFLOADING=1`, which moves the DiT to
CPU before the snapshot VAE decode. `SAMPLE_WITH_OFFLOADING=0` keeps it resident, which is
faster when VRAM allows.

## Model Paths

The wrapper defaults to Pyro's local Comfy model layout:

```text
DIT   /home/pyro/models/comfy/diffusion_models/qwen_image_2.1_int8_convrot.safetensors
TENC  /home/pyro/models/comfy/text_encoders/qwen3vl_8b_int8_convrot.safetensors
VAE   /home/pyro/models/comfy/vae/qwen_image_2.1_vae_bf16.safetensors
```

The bf16 Comfy text encoder works as well. The DiT must be a Comfy file with the fused
`img_mlp.gate_up` layout (the INT8 ConvRot release is).
