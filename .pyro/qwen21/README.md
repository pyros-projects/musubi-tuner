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

The run name is `${NAME}-${LORA_CONFIG}-${OPTIMIZER}`, plus `-diffsynth` and `-dedistill-<adapter>`
suffixes (see Timestep Sampling and Training Adapter), and the run writes to:

```text
/home/pyro/models/_out/qwen21/${RUN_NAME}
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

## Training Adapter (De-Distillation)

Plain LoRA training erodes the base model's ability to render without guidance: at CFG 1
the LoRA's samples get fused, ghosted bodies, while CFG 4 still renders them cleanly. The
community fix (Fizgig, SimpleTuner) is a frozen "training assistant" LoRA that sits on the
DiT during training only, so the trained LoRA learns the concept and not the drift. It is on
by default:

```text
TRAIN_LORA_OVERLAY  /home/pyro/models/comfy/loras/qwen21/training_adapter/fizgig_qwen_image_2.1_training_adapter.safetensors
```

The adapter:

- runs in every training forward, unmerged, at strength 1.0, with no gradients;
- is switched off for snapshots, so previews show base + trained LoRA (+ turbo) exactly as
  ComfyUI will;
- is never saved: checkpoints contain only the trained LoRA (its path is recorded in the
  metadata).

Two adapters are downloaded:

```text
fizgig_qwen_image_2.1_training_adapter.safetensors             rank 16, attention + MLP (default)
simpletuner_qwen_image_2.1_training_assistant_v2.safetensors   rank 32, attention only
```

Switch to SimpleTuner's, set a strength (`PATH:STRENGTH`), or train without one:

```bash
NAME=hs TRAIN_LORA_OVERLAY=/home/pyro/models/comfy/loras/qwen21/training_adapter/simpletuner_qwen_image_2.1_training_assistant_v2.safetensors .pyro/qwen21/train.sh
NAME=hs TRAIN_LORA_OVERLAY=/home/pyro/models/comfy/loras/qwen21/training_adapter/fizgig_qwen_image_2.1_training_adapter.safetensors:0.8 .pyro/qwen21/train.sh
NAME=hs TRAIN_LORA_OVERLAY= .pyro/qwen21/train.sh
```

Runs with an adapter get a `-dedistill-<adapter>` suffix on the run name, taken from the first
word of the file name (`-dedistill-fizgig`, `-dedistill-simpletuner`), so runs with different
adapters or none never resume each other.

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
default   main blocks attention plus MLP (192 layers), DiffSynth-Studio's default targets
preset-1  main blocks attention only: to_q, to_k, to_v, to_out.0 (128 layers)
preset-2  main blocks attention plus MLP: gate_up, out (192 layers, same as default)
preset-3  diffusion-pipe-ish: any Linear module with blocks. in its path (same 192 as default)
custom    use raw NETWORK_ARGS (within the blocks)
```

Unlike Krea2's "all Linears", nothing outside the blocks is trained. Qwen-Image 2.1 has one
modulation Linear shared by every block, which also modulates the text tokens; a LoRA there
rewrites how the whole prompt is read and broke composition (fused, ghosted bodies). The
embedders, norm_out and proj_out stay frozen as well, as in DiffSynth-Studio (Qwen's own
trainer) and diffusers.

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

## Timestep Sampling

`TIMESTEP_PRESET` picks how training noise levels are drawn:

```text
shift      default: sigmoid(randn), shifted by the official scheduler's resolution-dependent mu
diffsynth  DiffSynth-Studio (Qwen's own trainer): a uniform draw from its 1000-entry table
           (fixed mu 0.8, sigma 1.0 -> 0.02) with its bell-shaped loss weight (peak near sigma 0.5,
           zero at sigma 1, mean 1)
```

```bash
NAME=hs TIMESTEP_PRESET=diffsynth .pyro/qwen21/train.sh
```

`diffsynth` runs get a `-diffsynth` suffix on the run name, so they never resume a `shift` run.

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
