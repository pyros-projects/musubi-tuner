# Phase 3

## Goal

Add native `Z-Image` support and use it to validate which abstractions are truly shared versus `Flux2`-specific.

## Why This Phase Was Too Big

`Z-Image` is not just “Flux2 again.” Loader behavior, prompt handling, and LoRA normalization all add architecture-specific complexity. This needs multiple smaller passes.

## Execution Slices

### Phase 3A: Native Z-Image Loader and Initialization

**Deliverable**

Northstar can construct the native `Z-Image` model stack and finish startup successfully.

**Scope**

- create `northstar/models/zimage/`
- port model loading and architecture-specific setup
- support current local checkpoint paths

**Verification**

- startup completes for a `Z-Image` smoke config
- required model parts are initialized correctly

### Phase 3B: Native Z-Image Train Step

**Deliverable**

Northstar can run a one-step native `Z-Image` training smoke test.

**Scope**

- port the train step
- preserve conditioning behavior
- preserve current loss semantics

**Verification**

- one-step eager `Z-Image` train smoke run
- loss is finite
- optimizer step completes

### Phase 3C: Native Z-Image Sampling and LoRA

**Deliverable**

Northstar supports `Z-Image` sampling, LoRA training, and sampling LoRA normalization.

**Scope**

- port sampling path
- port prompt encoding path
- port LoRA training integration
- port Z-Image-specific sampling LoRA normalization

**Verification**

- sampling-only smoke run
- LoRA-enabled smoke run
- sampling LoRA affects output in the expected direction

### Phase 3D: Abstraction Pressure Test and Cleanup

**Deliverable**

Shared Northstar abstractions are revised so they serve both `Flux2` and `Z-Image` honestly.

**Scope**

- review common runtime hooks introduced during Phase 2
- split fake abstractions that only fit `Flux2`
- keep only the interfaces that both model families actually share

**Verification**

- both native `Flux2` and native `Z-Image` smoke runs still pass
- no model family depends on architecture-specific hacks hidden in shared code

## Exit Criteria

- Both target architectures run natively.
- Shared layers now reflect real commonality instead of optimistic guessing.
