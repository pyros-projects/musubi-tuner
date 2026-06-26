# Boogu Image Training Helper

`.pyro/boogu/train.sh` wraps the current Boogu Image Base LoRA workflow.

Run from the repository root:

```bash
CACHE_DATASET=1 SMOKE=1 .pyro/boogu/train.sh
```

## Defaults

```text
DIT  /home/pyro/models/comfy/diffusion_models/boogu_image_base_bf16.safetensors
VAE  /home/pyro/models/comfy/vae/ae.safetensors
TENC /home/pyro/models/comfy/text_encoders/qwen3vl_8b_fp8_scaled.safetensors
PROCESSOR /home/pyro/models/qwen3-vl-4b
```

Override them for RunPod or another machine:

```bash
DIT=/workspace/models/boogu_image_base_bf16.safetensors \
VAE=/workspace/models/ae.safetensors \
TENC=/workspace/models/qwen3vl_8b_fp8_scaled.safetensors \
PROCESSOR=/workspace/models/qwen3-vl-processor \
BOOGU_NAME=smoke .pyro/boogu/train.sh
```

## Cache Control

Set `CACHE_DATASET=1` to run both cache scripts before training:

```bash
BOOGU_NAME=smoke CACHE_DATASET=1 .pyro/boogu/train.sh
```

## Smoke Vs Longer Run

The `smoke` config defaults to a one-step smoke run and uses the local Boogu Turbo LoRA for preview sampling:

```bash
CACHE_DATASET=1 SMOKE=1 .pyro/boogu/train.sh
```

Because `SMOKE=auto` is the default, this shorter form is equivalent unless you override `BOOGU_NAME`:

```bash
CACHE_DATASET=1 .pyro/boogu/train.sh
```

`USE_TURBO_LORA=auto` means enabled for smoke runs and disabled for longer runs. Override it explicitly when needed:

```bash
SMOKE=1 USE_TURBO_LORA=0 .pyro/boogu/train.sh
BOOGU_NAME=lora_template USE_TURBO_LORA=1 .pyro/boogu/train.sh
```

For a longer template run:

```bash
BOOGU_NAME=lora_template MAX_STEPS=2000 SAVE_EVERY=100 SAMPLE_EVERY=50 .pyro/boogu/train.sh
```

Boogu block swap is not wired yet. The helper uses `--fp8_base --fp8_scaled` and gradient checkpointing for the first memory-saving path.

`NAME=...` is still honored only when it matches an existing `.pyro/boogu/cfg/*.toml` pair. This avoids accidental host-level `NAME` environment variables selecting nonexistent configs.
