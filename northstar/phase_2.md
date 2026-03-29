# Phase 2

## Goal

Replace Musubi-backed `Flux2` execution with a native Northstar `Flux2` runtime while keeping the config surface stable.

## Why This Phase Was Too Big

The original version bundled model loading, train step, sampling, LoRA, and compile behavior into one pass. That is too risky for a first native model port.

## Execution Slices

### Phase 2A: Native Flux2 Loader and Initialization

**Deliverable**

Northstar can construct the native `Flux2` model stack and finish startup successfully without beginning training.

**Scope**

- create `northstar/models/flux2/`
- port or wrap:
  - DiT loading
  - VAE loading
  - text encoder loading
  - dtype/device setup
- support local safetensors paths

**Verification**

- startup completes for a `Flux2` smoke config
- model components load on the expected device/dtype
- no training step is required yet

### Phase 2B: Native Flux2 Train Step

**Deliverable**

Northstar can run a one-step native `Flux2` training smoke test.

**Scope**

- port timestep/noise preparation
- port loss construction
- return optimizer-ready loss outputs
- match current semantics for:
  - timestep sampling
  - weighting scheme

**Verification**

- one-step eager `Flux2` train smoke run
- loss is finite
- optimizer step completes

### Phase 2C: Native Flux2 Sampling and LoRA

**Deliverable**

Northstar supports `sample_at_first`, periodic sampling, LoRA training, and sampling LoRA for eager `Flux2`.

**Scope**

- port prompt-to-sample flow
- support inline prompts
- port LoRA training integration
- port sampling LoRA handling for eager mode

**Verification**

- `sample_at_first` works
- periodic sampling works
- LoRA-enabled smoke run works
- sampling LoRA changes the sample output

### Phase 2D: Flux2 Compile, Prewarm, and FP8 Edge Cases

**Deliverable**

Northstar supports compiled `Flux2` training and the known-good sampling behavior around compile and FP8.

**Scope**

- port compile hooks
- port compile prewarm
- port compile-aware sampling LoRA behavior
- preserve FP8 fallback behavior during sampling where required

**Verification**

- compile smoke run
- compile + prewarm smoke run
- compile + sampling LoRA smoke run
- compile + FP8 + sampling LoRA smoke run

## Exit Criteria

- Native `Flux2` works in progressively richer modes without a single oversized implementation jump.
- The config shape remains stable across all slices.
