# LTX Prompt-File Image Pool Design

## Goal

Extend LTX prompt-file sampling so image-to-video inputs can be declared with the same inherited-root ergonomics as prompt-file LoRAs.

The feature should let a prompt TOML define:

- shared image sources at the root `[prompt]` level
- per-subset explicit image overrides
- per-subset plain T2V prompts with no image at all

This should work for:

- standalone inference via `python -m musubi_tuner.ltx2_generate_video --sample_prompts ...`
- training preview sampling via `ltx2_train_network.py` / `ltx2_train.py`
- latent precache flows that already read `image_path` from sample prompts

The design goal is to keep the runtime sampling path simple by resolving inheritance and assignment during prompt-file loading, not during denoising.

## User Intent

The desired experience is:

- root image inputs behave like root LoRAs: one declaration, many prompt subsets inherit it
- `input_images_dir = "..."` should work as a prompt-file-native image pool
- `image_input_order = "random"` should assign a random image from that pool to each eligible subset
- some subsets should still remain plain T2V prompts without special ceremony

This is not a Cartesian-product compare mode. The user wants one resolved prompt entry per subset, with prompt-local image assignment as a convenience layer.

## Current State

Today:

- LTX prompt TOML supports root defaults plus per-subset overrides.
- Shared prompt parsing already supports `--i <path>` in `.txt` prompt files, which becomes `image_path`.
- LTX sample execution already supports I2V when a sample parameter contains `image_path`.
- LTX latent precache already stores `conditioning_latent` keyed by prompt index, so prompt-level `image_path` is the existing integration point.

What is missing is prompt-file-native image inheritance and pool assignment for TOML.

## Design Principles

- Reuse the existing `image_path` runtime contract.
- Keep prompt-file behavior parallel to prompt-file LoRA inheritance where possible.
- Fail clearly when assignment cannot be resolved.
- Keep plain T2V as a first-class, obvious path.
- Avoid introducing a second i2v-specific DSL if simple prompt normalization can do the job.

## Approaches Considered

### 1. Resolver-based inherited image pool

Add root-level image source fields to the prompt TOML. During prompt-file resolution, assign each eligible subset exactly one `image_path` and emit ordinary resolved prompt dicts.

Pros:

- clean mental model
- no sampler changes beyond consuming resolved `image_path` as today
- fits the existing root-default plus subset-override pattern
- easy to test at the prompt resolver layer

Cons:

- requires a bit more resolver logic
- pool assignment order becomes part of prompt-file semantics

### 2. Runtime pool assignment inside sampling loop

Keep the prompt resolver dumb and let the sampling runtime decide which image to use for each subset just before execution.

Pros:

- fewer prompt-loader changes

Cons:

- pushes config semantics into runtime code
- makes precache flows harder to reason about
- creates more hidden state around prompt order and randomness

### 3. Expand subsets into prompt × image combinations

Treat image pools as generators that duplicate prompt subsets into multiple resolved samples.

Pros:

- useful for compare-mode or bulk sweeps

Cons:

- not what the user asked for
- turns one prompt subset into many outputs
- complicates output count, seeds, cache behavior, and documentation

## Recommendation

Use approach 1: resolver-based inherited image pools.

It keeps the runtime boring and predictable. The prompt-file resolver becomes the single place that knows how to turn friendly root-level config into plain resolved per-sample fields:

- `image_path`
- or no `image_path` for T2V

That keeps denoising, latent precache, and training preview aligned around one existing field instead of proliferating special cases.

## Proposed Prompt File Shape

### Root-level image sources

The root `[prompt]` table may define either:

- `input_images = ["/abs/a.png", "/abs/b.png"]`
- `input_images_dir = "/abs/path/to/images"`
- or both

If both are present, they are combined into one candidate pool.

Optional root image controls:

- `image_input_order = "sorted" | "random"`
- `image_assignment = "unique"`

Example:

```toml
[prompt]
width = 1280
height = 832
frame_count = 49

loras = [
  { path = "/models/base_style.safetensors", weight = 0.35, merge = false },
]

input_images_dir = "/data/shot_stills/forest_set"
image_input_order = "random"
image_assignment = "unique"
```

### Per-subset image fields

Each `[[prompt.subset]]` may define one of:

- `image_path = "/abs/specific.png"` to force I2V for that subset
- `use_image_pool = true` to opt into root-pool assignment explicitly
- `use_image_pool = false` to force plain T2V even if a root pool exists

Example:

```toml
[[prompt.subset]]
prompt = "slow handheld walk through a mossy forest at dawn"
use_image_pool = true

[[prompt.subset]]
prompt = "wolf emerging between trees, cinematic realism"
image_path = "/data/special_cases/wolf_hero.png"

[[prompt.subset]]
prompt = "wide aerial shot over pine canopy and river valley"
use_image_pool = false
```

## Effective Semantics

The resolver should produce exactly one of these outcomes per subset:

1. explicit `image_path`
2. inherited pool-assigned `image_path`
3. no `image_path` at all, meaning plain T2V

### Eligibility rules

For each subset:

- if `image_path` is set, use it and skip pool assignment
- else if `use_image_pool = false`, leave it as T2V
- else if `use_image_pool = true`, assign from the root pool
- else if a root pool exists, assign from the root pool by default
- else leave it as T2V

This gives the user a smooth default while still allowing prompt-local opt-out.

### Why default eligible when a root pool exists

Root LoRAs already work as inherited defaults. Matching that pattern for images makes the config feel coherent:

- root `loras` apply unless replaced
- root image pool applies unless opted out or explicitly overridden

The explicit `use_image_pool = false` field makes T2V intent readable without requiring dummy values like `image_path = ""`.

## Pool Construction

The candidate pool is built from:

- all entries in `input_images`
- all supported image files recursively found under `input_images_dir`

Supported files should reuse the same image extension policy already accepted by existing PIL-based sampling paths. The resolver should ignore non-image files.

If `image_input_order = "sorted"`:

- sort by normalized path string for deterministic behavior

If `image_input_order = "random"`:

- shuffle once per prompt-file resolution pass

Random ordering should use Python's standard RNG and should not be tied to denoising seeds.

## Assignment Policy

V1 supports one assignment mode:

- `image_assignment = "unique"`

Meaning:

- each pool-assigned subset receives one distinct image
- explicit `image_path` subsets do not consume from the pool
- T2V subsets do not consume from the pool

If there are fewer available pool images than eligible subsets, fail with a clear error.

Example error:

```text
Prompt file image pool under [prompt] resolved 3 images, but 5 subsets require pool assignment.
Add more images, mark some subsets with use_image_pool = false, or provide explicit image_path values.
```

We intentionally do not add silent cycling or reuse in v1. Reuse can be added later as a new explicit assignment mode if needed.

## Plain T2V Support

Plain T2V is supported in two ways:

### 1. No root image pool

If the root prompt has no `input_images`, no `input_images_dir`, and the subset has no `image_path`, the subset is ordinary T2V.

### 2. Explicit T2V opt-out under a root pool

If a root pool exists but a subset should remain T2V, set:

```toml
use_image_pool = false
```

This is the recommended readable escape hatch.

## Architecture

### 1. Prompt-file resolver owns inheritance and assignment

Extend the LTX TOML resolver so it:

- parses root image source fields
- enumerates root candidate images
- resolves subset eligibility
- assigns pool images when needed
- emits final prompt dicts with ordinary `image_path` fields

The runtime sampler should not know about:

- `input_images`
- `input_images_dir`
- `image_input_order`
- `image_assignment`
- `use_image_pool`

Those are prompt-file authoring conveniences only.

### 2. Sampling runtime remains unchanged in shape

Sampling already knows how to do:

- I2V if `sample_parameter["image_path"]` exists
- T2V otherwise

That contract should remain the same.

### 3. Latent precache follows resolved prompts

Any latent precache flow that already reads prompt-derived `image_path` should work automatically once the resolver emits final per-subset `image_path` values.

## Data Flow

### Standalone inference

1. Parse CLI args
2. Load LTX prompt TOML
3. Resolve root defaults, LoRAs, and image pool assignment
4. Produce resolved prompt dicts
5. Continue with existing sample prompt processing
6. Sampler sees either `image_path` or nothing

### Training preview sampling

1. Load LTX prompt TOML
2. Resolve root defaults, LoRAs, and image pool assignment
3. Produce resolved prompt dicts
4. Continue with existing preview prompt processing
5. Preview sampler sees either `image_path` or nothing

## Validation Rules

The resolver should validate:

- `input_images` is a list of non-empty strings when present
- `input_images_dir` is a non-empty string when present
- `image_input_order` is one of `sorted`, `random`
- `image_assignment` is `unique` in v1
- `use_image_pool` is boolean when present
- explicit `image_path` values are non-empty strings

File validation:

- error if `input_images_dir` does not exist
- error if explicit `image_path` does not exist
- error if a listed `input_images` path does not exist
- error if root pool resolves to zero images but some subsets require pool assignment
- error if unique assignment cannot satisfy all eligible subsets

## Precedence Rules

Highest to lowest:

1. explicit subset `image_path`
2. explicit subset `use_image_pool = false`
3. explicit subset `use_image_pool = true`
4. inherited root pool default
5. plain T2V with no image

Notes:

- explicit `image_path` always wins
- `use_image_pool = false` only matters when `image_path` is absent
- root pool settings never override an explicit subset decision

## Documentation Plan

Update:

- `docs/ltx_2.md`
- `docs/config_examples/ltx2_sample_prompts_two_stage.toml`

Docs should show:

- pure T2V subsets
- pool-assigned I2V subsets
- explicit subset `image_path`
- root-level image pool with `image_input_order = "random"`

## Testing Plan

Add focused resolver tests covering:

- root pool assignment in sorted order
- root pool assignment in random order
- explicit subset `image_path` bypassing the pool
- `use_image_pool = false` leaving a subset as T2V
- mixed prompt file with T2V and I2V subsets
- not enough images for unique assignment
- invalid root image config values
- nonexistent files and empty directories

Add at least one integration-oriented test verifying that resolved prompts fed into LTX sample processing contain:

- ordinary `image_path` for I2V subsets
- no `image_path` for T2V subsets

## Out of Scope

- image reuse or cycling modes
- prompt × image Cartesian expansion
- compare-mode batch generation
- prompt-local `input_images_dir`
- video pools for V2V in this same change
- GUI support

Those may be useful later, but they are not needed for a clean v1.

## Open Questions Resolved

### Should regular T2V prompts still be allowed?

Yes. This is an explicit requirement.

### Should pool assignment silently reuse images if the pool is too small?

No. V1 should fail fast and ask the user to be explicit.

### Should the runtime sampler learn image-pool semantics?

No. The resolver should translate friendly config into plain resolved `image_path` fields.
