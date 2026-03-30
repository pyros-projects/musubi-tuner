# LTX Prompt-File Per-Prompt LoRA Design

## Goal

Enable LTX prompt files to define per-prompt LoRA stacks directly in `--sample_prompts` TOML, so prompt A can use one concept/style set, prompt B can use another, and prompt C can fall back to a shared baseline.

This should work for:

- standalone inference via `python -m musubi_tuner.ltx2_generate_video --sample_prompts ...`
- training preview sampling via `ltx2_train_network.py` / `ltx2_train.py`

The feature is intentionally prompt-file-native. We are not designing a second compare-mode DSL or a separate interactive LoRA system here.

## Current State

Today, LTX prompt files already support per-prompt scalar overrides like:

- `width`
- `height`
- `frame_count`
- `sample_steps`
- `sample_sigmas`
- `guidance_scale`
- `stage1_distilled_lora_multiplier`
- `stage2_distilled_lora_multiplier`

Standalone LTX inference still applies CLI `--lora_weight/--lora_multiplier` once before the prompt loop, which makes those LoRAs global to the whole run.

The shared sampling runtime, however, already supports reversible sampling LoRA application:

- runtime overlays for standard Linear LoRA modules on quantized transformers
- eager-merge fallback with restoration when overlays are not possible
- explicit cleanup/restoration hooks between sampling phases

LTX two-stage inference already uses this runtime path for distilled LoRA application and stage-specific switching, so the model surface is capable of sample-local LoRA changes.

## Proposed Prompt File Shape

### Root-level baseline LoRAs

The root `[prompt]` table may define a baseline stack applied to every prompt by default:

```toml
[prompt]
width = 1280
height = 832
frame_count = 49

loras = [
  { path = "/models/loras/base_style.safetensors", weight = 0.35, merge = false },
  { path = "/models/loras/camera_motion_helper.safetensors", weight = 0.20, merge = false }
]
```

### Per-subset prompt LoRAs

Each `[[prompt.subset]]` entry may define:

- `loras = [...]`
- `lora_mode = "extend" | "replace"`

Example:

```toml
[[prompt.subset]]
prompt = "cinematic close-up of a woman walking through neon rain at night"
seed = 1234
lora_mode = "extend"
loras = [
  { path = "/models/loras/concept_a.safetensors", weight = 0.85, merge = false },
  { path = "/models/loras/concept_b.safetensors", weight = 0.60, merge = false }
]

[[prompt.subset]]
prompt = "two tennis players rallying on an outdoor clay court"
seed = 2345
lora_mode = "replace"
loras = [
  { path = "/models/loras/concept_c.safetensors", weight = 0.90, merge = false }
]
```

### Effective semantics

- If a subset omits `loras`, it uses only the root `[prompt].loras` baseline.
- If a subset sets `lora_mode = "extend"`, effective LoRAs are:
  - root baseline LoRAs
  - followed by subset LoRAs
- If a subset sets `lora_mode = "replace"`, effective LoRAs are:
  - subset LoRAs only
- If `lora_mode` is omitted and subset `loras` exist, default to `"extend"`.

This keeps behavior explicit and readable inside the TOML itself.

## LoRA Entry Schema

Each LoRA entry is a TOML inline table:

```toml
{ path = "/abs/path/to/lora.safetensors", weight = 0.8, merge = false }
```

### Required fields

- `path`: absolute or repo-relative LoRA path

### Optional fields

- `weight`: float, default `1.0`
- `merge`: bool, default `false`

### Notes

- `merge = false` is the recommended path for prompt-file LoRAs because it aligns with reversible runtime overlays.
- `merge = true` is allowed for compatibility, but it should be treated as slower and more fragile because it requires eager restore behavior around each sample.
- We are not adding block-filtering or extra adapter metadata in v1.

## Scope Boundary

### In scope

- TOML prompt-file support for root and per-subset `loras`
- explicit per-subset `lora_mode`
- standalone LTX inference
- LTX training preview sampling
- reuse of cached LoRA state dicts by path
- per-sample application and restoration of the effective LoRA stack

### Out of scope

- compare-mode combinatorics like `group_union` or `group_product`
- automatic Cartesian products of LoRA choices
- GUI/editor support
- non-LTX prompt-file upgrades in the same change
- block-level LoRA routing extensions

Those can come later if this direct prompt-file path proves useful.

## Architecture

### 1. Prompt loading

Keep `load_prompts()` as the basic TOML loader, but allow it to carry `loras` and `lora_mode` fields through unchanged.

The loader should not try to apply semantics. It should only preserve the raw prompt dictionaries.

### 2. Prompt normalization

LTX prompt processing should normalize each prompt dict into a resolved per-sample LoRA spec:

- parse root `[prompt].loras`
- parse subset `loras`
- resolve subset `lora_mode`
- build a final `resolved_loras` list for each sample

This resolved list should be stored inside each sample parameter dict so later sampling code does not need to know TOML inheritance rules.

### 3. Sampling application

Before each sample:

1. resolve the effective LoRA list from `sample_parameter["resolved_loras"]`
2. load/cache the referenced state dicts by path
3. apply the sample-local LoRA stack to the transformer
4. run single-stage or two-stage inference
5. restore the transformer to its clean baseline

This is the key architectural change: prompt-file LoRAs become per-sample state, not one global pre-loop merge.

### 4. Distilled LoRA separation

Prompt-file concept/style LoRAs remain separate from the existing distilled LoRA path:

- prompt-file LoRAs are general user-selected adapters
- distilled LoRA remains the dedicated two-stage refinement adapter

This prevents the new feature from tangling with the stage-2 control flow already implemented for LTX distilled sampling.

## Data Flow

### Standalone inference

1. Parse CLI args
2. Load prompt file
3. Normalize prompts into sample parameter dicts
4. Build `resolved_loras` per sample
5. Enter prompt loop
6. Apply prompt-local LoRAs
7. Run LTX inference
8. Restore LoRAs
9. Move to next prompt

### Training preview sampling

1. Load prompt file during sample prompt processing
2. Normalize prompts into sample parameter dicts
3. Batch-encode text embeddings as today
4. During the preview prompt loop, apply `resolved_loras` per sample
5. Run preview inference
6. Restore LoRAs
7. Continue to next prompt

## Precedence Rules

The system currently has several LoRA entry points:

- standalone CLI `--lora_weight/--lora_multiplier`
- generic sampling LoRA machinery in the shared trainer
- prompt-file-local `loras` proposed here
- distilled LoRA for two-stage refinement

For v1, precedence should be:

1. distilled LoRA stays independent and always behaves as it does today
2. prompt-file `resolved_loras` define the per-sample concept/style stack
3. CLI global inference LoRAs remain available, but should be treated as the baseline stack only

Recommended behavior:

- standalone CLI `--lora_weight/--lora_multiplier` should be folded into the root baseline semantics for standalone inference
- prompt-file `lora_mode = "replace"` should replace that baseline for the specific prompt
- training preview should use prompt-file `resolved_loras` for per-sample behavior and keep the older shared sampling LoRA path unchanged unless explicitly needed

This avoids surprising double-application while still preserving old CLI workflows.

## Error Handling

Validation should fail early and clearly for:

- `loras` is not a list
- a LoRA entry is not a table/dict
- missing `path`
- invalid `weight`
- invalid `merge`
- invalid `lora_mode`

Runtime warnings should be used for:

- a LoRA that matches no modules
- `merge = true` on prompt-file LoRAs
- mixed runtime-overlay and eager-merge fallback behavior inside one sample

If a prompt fails due to a broken LoRA file, existing per-prompt error recovery should skip that prompt and continue when possible.

## Performance Expectations

The main performance cost is no longer model reload; it is per-sample adapter application/restoration.

Expected behavior:

- `merge = false` with standard Linear LoRAs on NF4/FP8 models should mostly use runtime overlays
- repeated prompt reuse of the same LoRA path should benefit from cached loaded state dicts
- `merge = true` or unsupported module shapes may still incur eager merge/restore overhead

This is acceptable for prompt-file-driven creative sampling, especially because it unlocks much more expressive per-prompt control without reloading the whole transformer.

## Testing Plan

### Parsing tests

- root `[prompt].loras` survives load
- subset `loras` survives load
- subset `lora_mode` survives load

### Resolution tests

- no subset `loras` => baseline only
- subset `extend` => baseline + subset
- subset `replace` => subset only
- omitted `lora_mode` with subset `loras` => `extend`

### Sampling tests

- prompt A and prompt B can use different LoRA stacks without state leaking across prompts
- prompt-level `merge = false` uses runtime overlay path when supported
- prompt-level fallback merge restores cleanly when overlays are unsupported

### Two-stage tests

- prompt-file concept/style LoRAs coexist with distilled stage-2 LoRA switching
- prompt-local LoRAs do not corrupt stage-specific distilled LoRA multiplier behavior

## Recommended v1 Implementation Strategy

1. Extend prompt normalization to produce `resolved_loras`
2. Add a small helper that applies/restores a sample-local LoRA list from a prompt dict
3. Wire that helper into the standalone LTX prompt loop
4. Wire the same helper into LTX training preview sampling
5. Add docs and prompt-file examples
6. Add focused tests before expanding further

This keeps the change extraction-friendly and close to the current Musubi runtime behavior.

## Recommendation

Ship the minimal direct version first:

- prompt-file-native `loras`
- explicit `lora_mode`
- per-sample apply/restore
- no compare combinatorics

That gives Pyro the exact workflow requested:

- prompt A with Concept A + Concept B
- prompt B with Concept C
- prompt C with just the baseline stack

without turning LTX prompt files into a second inference framework.
