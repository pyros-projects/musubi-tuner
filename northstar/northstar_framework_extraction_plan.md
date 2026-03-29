# Northstar Framework Extraction Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a clean, config-first training framework for `Flux2` and `Z-Image` from Musubi while preserving Musubi's runtime behavior and performance characteristics where they matter.

**Architecture:** Build a small core runtime with typed TOML config loading, a plugin boundary for model families, and a parity harness that treats the current Musubi implementation as the behavioral reference. Port `Flux2` first, then `Z-Image`, and only generalize abstractions that survive both ports without harming performance.

**Tech Stack:** Python, PyTorch, Accelerate, TOML, safetensors, schema validation, Musubi model code where reuse is still the fastest and safest path.

---

## Operator UX Target

- One self-contained run file, for example: `my_run.toml`
- Launch shape:
  - `accelerate launch train.py my_run.toml`
- No separate:
  - dataset config file
  - sample prompt file
  - mandatory cache-preparation commands
  - giant startup-argument list
- The run file should be able to represent:
  - everything currently expressed by the trainer CLI
  - everything currently expressed in dataset TOMLs
  - everything currently expressed in prompt files used for sampling

### Example Intent

The following current launch style:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/flux_2_train_network.py \
  --model_version klein-base-9b \
  --dit /home/pyro/models/comfy/diffusion_models/flux-2-klein-base-9b.safetensors \
  --vae /home/pyro/models/ae.safetensors \
  --text_encoder /home/pyro/models/flux-klein-base-9b/text_encoder/model-00001-of-00004.safetensors \
  --sdpa --mixed_precision bf16 \
  --timestep_sampling flux2_shift --weighting_scheme none \
  --optimizer_type adafactor \
  --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False" \
  --lr_scheduler constant \
  --learning_rate 2e-4 \
  --max_grad_norm 0 \
  --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --network_module networks.lora_flux_2 --network_dim 16 --network_alpha 16 \
  --max_train_epochs 60 --save_every_n_epochs 1 --seed 42 --sample_at_first --sample_every_n_steps 50 \
  --output_dir /home/pyro/models/_out/bb_flux --output_name bb_flux --sample_prompts data/prompts_bb_flux.txt \
  --dataset_config data/dataset_bb_flux.toml --fp8_base --fp8_scaled --logging_dir /home/pyro/models/_out/bb_flux \
  --sampling_lora_weight /home/pyro/models/comfy/loras/flux2_9/klein_base_to_turbo_all_comfy.safetensors \
  --sampling_lora_multiplier 0.6 --compile_dynamic true --cuda_allow_tf32 --compile_prewarm
```

should become something like:

```bash
accelerate launch train.py my_run.toml
```

with `my_run.toml` containing:
- training config
- model paths
- optimizer config
- performance config
- dataset definition
- sampling prompt definitions
- sampling LoRA definitions
- output/logging config

and the training pipeline should automatically perform prerequisite cache work when needed, instead of requiring separate commands like:

```bash
python src/musubi_tuner/flux_2_cache_latents.py ...
python src/musubi_tuner/flux_2_cache_text_encoder_outputs.py ...
```

## Scope

- In scope:
  - `Flux2`
  - `Z-Image`
  - train
  - sample-during-training
  - LoRA training
  - sampling LoRA
  - compile / prewarm
  - dataset and cache integration needed for those two models

- Out of scope for the first roadmap:
  - every other Musubi architecture
  - GUI
  - complete Diffusers-native runtime
  - broad refactors for all existing docs and scripts

## Working Assumptions

- Musubi is the current reference implementation, even when its structure is messy.
- Performance regressions are unacceptable in the hot path.
- Typed TOML config is a major goal, not an afterthought.
- Each phase must produce a runnable artifact you can test directly.

## Proposed Future Layout

- `northstar/config/`
  - schema definitions
  - TOML loading
  - validation
  - migration helpers
  - single-file run schema
- `northstar/core/`
  - run context
  - training loop primitives
  - logging
  - checkpointing
  - compile integration
- `northstar/data/`
  - dataset graph
  - bucketing
  - cache readers
  - sampling prompt loading
- `northstar/models/flux2/`
  - loader
  - train step
  - sample step
  - LoRA normalization hooks
- `northstar/models/zimage/`
  - loader
  - train step
  - sample step
  - LoRA normalization hooks
- `northstar/interop/`
  - Musubi compatibility adapters
  - optional PEFT / Diffusers bridges
- `examples/`
  - single-file smoke configs
  - single-file parity configs
  - minimal local experiments
- `tests/`
  - unit tests
  - parity tests
  - smoke checks

## Delivery Strategy

- Do not start by rewriting everything.
- Start by building a clean shell around known-good Musubi behavior.
- Treat parity tooling as a first-class feature.
- Only replace Musubi internals when a phase already has a runnable fallback.
- Collapse config sprawl into one run TOML early, not late.

## Lessons From diffusion-pipe

The companion project at `/home/pyro/projects/private/diffusion-pipe` already demonstrates several patterns that are worth carrying into Northstar.

### Keep

- Config-first operation.
  - `diffusion-pipe` already treats TOML as the main operator interface via `train.py --config ...`.
  - Northstar should push this one step further by collapsing training, dataset, prompts, and cache policy into a single run TOML.

- Model-family plugin boundaries.
  - `diffusion-pipe` has a clearer model abstraction split through files like `models/base.py`, `models/flux_2_klein.py`, and `models/z_image.py`.
  - Northstar should preserve this idea: each model family gets its own loader, train step, sample step, and adapter normalization hooks.

- Cache orchestration as part of the training system.
  - `diffusion-pipe` already centralizes cache behavior in the training pipeline and dataset layer.
  - Northstar should keep that spirit but improve UX so cache preparation is automatic by default.

- Separate core runtime from utility tools.
  - Conversion scripts, inference helpers, and analysis tools should live outside the hot training runtime.
  - This keeps the main framework easier to reason about.

- Example-driven ergonomics.
  - The large number of working config examples in `diffusion-pipe` is a real strength.
  - Northstar should ship a small but high-quality set of example run TOMLs early.

### Borrow Carefully

- PEFT-backed adapter abstractions.
  - `diffusion-pipe` shows that PEFT can be useful at the adapter boundary.
  - Northstar can borrow this for config and interoperability, but should not assume PEFT owns the hot path if that costs performance or flexibility.

- Diffusers conventions.
  - Config and model packaging conventions from Diffusers can reduce reinvention.
  - Use them only where they do not compromise runtime behavior or performance.

### Do Not Copy Blindly

- Deepspeed pipeline-parallel architecture as the foundation.
  - This is valuable for `diffusion-pipe`, but it is not automatically the right center of gravity for Northstar.
  - Northstar should choose its runtime core based on `Flux2` and `Z-Image` needs first.

- Split config UX.
  - `diffusion-pipe` still commonly separates main config and dataset config.
  - Northstar should explicitly improve on this with a single run file.

- Giant monolithic orchestration files.
  - `diffusion-pipe/train.py` is still a large central script.
  - Northstar should prefer smaller focused runtime units once the initial shell is working.

### Northstar Synthesis

The target is not “Musubi but cleaner” or “diffusion-pipe but for different models.”

The target is:

- Musubi's proven `Flux2` and `Z-Image` behavior
- diffusion-pipe's config-first operator philosophy
- a smaller and more explicit internal architecture than either codebase currently has

## Single-File TOML Shape

The framework should converge on one primary document per run:

```toml
[run]
name = "bb_flux"
architecture = "flux2"
seed = 42

[accelerate]
num_cpu_threads_per_process = 1
mixed_precision = "bf16"

[model]
version = "klein-base-9b"
dit = "/home/pyro/models/comfy/diffusion_models/flux-2-klein-base-9b.safetensors"
vae = "/home/pyro/models/ae.safetensors"
text_encoder = "/home/pyro/models/flux-klein-base-9b/text_encoder/model-00001-of-00004.safetensors"
sdpa = true
fp8_base = true
fp8_scaled = true

[training]
timestep_sampling = "flux2_shift"
weighting_scheme = "none"
gradient_checkpointing = true
prepare_cache_if_needed = true
learning_rate = 2e-4
max_grad_norm = 0
max_train_epochs = 60
save_every_n_epochs = 1
max_data_loader_n_workers = 2
persistent_data_loader_workers = true

[optimizer]
type = "adafactor"
lr_scheduler = "constant"

[optimizer.args]
scale_parameter = false
relative_step = false
warmup_init = false

[network]
module = "networks.lora_flux_2"
dim = 16
alpha = 16

[performance]
compile_dynamic = true
compile_prewarm = true
cuda_allow_tf32 = true

[output]
dir = "/home/pyro/models/_out/bb_flux"
name = "bb_flux"
logging_dir = "/home/pyro/models/_out/bb_flux"

[sampling]
enabled = true
sample_at_first = true
sample_every_n_steps = 50

[[sampling.loras]]
path = "/home/pyro/models/comfy/loras/flux2_9/klein_base_to_turbo_all_comfy.safetensors"
multiplier = 0.6

[[sampling.prompts]]
prompt = "a pretty and skinny woman doing a backcatch pose at home in her room, wearing casual gothic clothing. amateur candid shot. The photo is mildly overexposed from flashlight."
width = 832
height = 1216
steps = 6
seed = 6666
guidance = 4
negative_prompt = " "

[[dataset.groups]]
resolution = [320, 320]
batch_size = 4
enable_bucket = true
bucket_no_upscale = false
cache_directory = "cache/backcatch/2press320"
caption_extension = ".txt"

[[dataset.groups.sources]]
image_directory = "/home/pyro/datasets/iterator/cont400/backcatch/2press"
num_repeats = 1

[[dataset.groups]]
resolution = [640, 640]
batch_size = 4
enable_bucket = true
bucket_no_upscale = false
cache_directory = "cache/backcatch/2press640"
caption_extension = ".txt"

[[dataset.groups.sources]]
image_directory = "/home/pyro/datasets/iterator/cont400/backcatch/2press"
num_repeats = 1
```

### TOML Design Rules

- The file must be hand-editable.
- Inline prompt definitions are first-class.
- Inline dataset definitions are first-class.
- Optional includes can be added later, but the primary UX is one file.
- Sections should map to operator intent, not internal implementation details.
- Unknown keys should fail validation.
- Deprecated keys should provide migration messages.
- Cache creation should be automatic by default when prerequisites are missing or stale.

## Cache Preparation UX

The new framework should absorb the work currently done by separate cache scripts, such as:

```bash
python src/musubi_tuner/flux_2_cache_latents.py \
  --dataset_config data/dataset_bb_flux.toml \
  --vae /home/pyro/models/ae.safetensors \
  --model_version klein-base-9b

python src/musubi_tuner/flux_2_cache_text_encoder_outputs.py \
  --dataset_config data/dataset_bb_flux.toml \
  --text_encoder /home/pyro/models/flux-klein-base-9b/text_encoder/model-00001-of-00004.safetensors \
  --batch_size 16 \
  --model_version klein-base-9b
```

Target behavior:

- `accelerate launch train.py my_run.toml` should inspect required caches before training starts.
- If caches are missing, it should build them automatically.
- If caches are stale relative to model/config inputs, it should either:
  - rebuild automatically, or
  - stop with a precise explanation, depending on policy
- The operator should not need to remember separate prep commands in normal usage.

Suggested policy:

- Default:
  - `prepare_cache_if_needed = true`
- Optional advanced controls:
  - `fail_if_cache_missing = false`
  - `rebuild_stale_cache = false`
  - `cache_only = false`

This keeps the common workflow simple while preserving explicit control for advanced users.

## Phase Overview

### Phase 0: Parity Harness and Config Contract

**Intent:** Create the safety rails before any extraction work.

**Runnable deliverable:**
- A small CLI that loads a single typed run TOML and can execute reference smoke runs through current Musubi code paths.
- Example:
  - `python -m northstar.cli.train examples/flux2_smoke.toml --backend musubi`

**What this phase proves:**
- The future config surface is not just a design sketch.
- You can already drive existing behavior through a cleaner config contract.
- You have a parity baseline for later phases.

**Core outputs:**
- `northstar/config/schema.py`
- `northstar/config/load.py`
- `northstar/interop/musubi_runner.py`
- `northstar/interop/musubi_cache_runner.py`
- `examples/flux2_smoke.toml`
- `examples/zimage_smoke.toml`
- `tests/test_config_validation.py`
- `tests/test_musubi_reference_runner.py`

**Execution slices:**

- **Phase 0A: Minimal schema and loader**
  - define the single-file top-level config sections
  - load and validate a minimal TOML
  - fail clearly on malformed or unknown keys

- **Phase 0B: Full config contract and examples**
  - expand the schema to include:
    - `sampling`
    - `dataset`
    - cache policy fields
  - add one reference smoke config per model family
  - support inline prompts and inline dataset definitions

- **Phase 0C: Musubi reference runner**
  - map single-file TOML fields to current Musubi arguments for `Flux2` and `Z-Image`
  - add the reference execution shim
  - optionally support import of old prompt files and dataset TOMLs for migration

- **Phase 0D: Cache shim and parity report**
  - add a reference cache-preparation shim
  - define the first parity report format:
    - sample image paths
    - run metadata
    - timing summary
    - compile summary

**How you can test it:**
- Run a one-step `Flux2` smoke config from a single TOML file.
- Run a one-step `Z-Image` smoke config from a single TOML file.
- Confirm the CLI validates configs and delegates correctly.
- Confirm a run can trigger prerequisite cache preparation automatically.

**Why this phase matters:**
- It gives immediate value before any risky rewrite.
- It prevents architecture drift by making config semantics explicit early.

---

### Phase 1: New Framework Skeleton With Musubi-Backed Runtime

**Intent:** Move orchestration into the new framework while still relying on Musubi internals for execution.

**Runnable deliverable:**
- A new `northstar` CLI that owns:
  - single-file run config parsing
  - run directory setup
  - metadata recording
  - logging
  - smoke orchestration
-  - automatic cache preparation
- Training still executes through Musubi-backed adapters.

**Core outputs:**
- `northstar/cli/train.py`
- `northstar/core/run_context.py`
- `northstar/core/logging.py`
- `northstar/core/metadata.py`
- `northstar/interop/musubi_flux2_adapter.py`
- `northstar/interop/musubi_zimage_adapter.py`

- [ ] Make the new CLI the primary entrypoint for smoke runs.
- [ ] Write config-to-runtime translation only once in the new framework.
- [ ] Capture run manifests in a stable machine-readable format.
- [ ] Remove the need for separate prompt and dataset files in normal operation.
- [ ] Remove the need for separate cache-preparation commands in normal operation.
- [ ] Standardize log summaries for:
  - dataset buckets
  - cache preparation decisions
  - compile / prewarm
  - sampling LoRA
  - checkpoint outputs

**How you can test it:**
- Launch both models through `northstar`, not through Musubi scripts directly.
- Compare outputs and runtime metadata against the Phase 0 reference.
- Confirm that one TOML file is all you need for a normal run.
- Confirm first-run cache preparation and second-run cache reuse both behave sensibly.

**Exit criteria:**
- The new CLI is already nicer to use than raw Musubi arguments.
- Changing config does not require editing shell scripts full of flags.

---

### Phase 2: Native Flux2 Runtime

**Intent:** Port `Flux2` into a clean native implementation inside the new framework.

**Runnable deliverable:**
- A native `Flux2` trainer and sampler in `northstar` that no longer depends on Musubi's `flux_2_train_network.py` for execution.
- Example:
  - `python -m northstar.cli.train --config examples/flux2_smoke.toml --backend native`

**Core outputs:**
- `northstar/models/flux2/loader.py`
- `northstar/models/flux2/train_step.py`
- `northstar/models/flux2/sample.py`
- `northstar/models/flux2/lora.py`
- `northstar/models/flux2/performance.py`
- `tests/flux2/test_native_flux2_smoke.py`
- `tests/flux2/test_flux2_sampling_lora.py`

- [ ] Reuse model-loading code where reuse is cheaper than translation.
- [ ] Port training-step behavior with parity tests.
- [ ] Port sample-during-training behavior.
- [ ] Port compile / prewarm behavior.
- [ ] Port sampling LoRA handling, including compile and FP8 fallback behavior.

**How you can test it:**
- Run one-step smoke training.
- Run `sample_at_first`.
- Run `compile + compile_prewarm`.
- Run `sampling_lora` with and without FP8.

**Exit criteria:**
- Native `Flux2` produces samples and logs that are close enough to the Musubi reference to be considered parity for smoke use.
- The config surface remains stable.

---

### Phase 3: Native Z-Image Runtime

**Intent:** Port `Z-Image` as the second model family and use that port to pressure-test abstractions.

**Runnable deliverable:**
- Native `Z-Image` support under the same CLI and config system.

**Core outputs:**
- `northstar/models/zimage/loader.py`
- `northstar/models/zimage/train_step.py`
- `northstar/models/zimage/sample.py`
- `northstar/models/zimage/lora.py`
- `tests/zimage/test_native_zimage_smoke.py`
- `tests/zimage/test_zimage_sampling_lora.py`

- [ ] Port the Z-Image-specific prompt encoding and sampling path.
- [ ] Port Z-Image sampling LoRA normalization behavior.
- [ ] Validate that shared abstractions from `Flux2` are actually reusable.
- [ ] Split any fake abstractions that only worked for `Flux2`.

**How you can test it:**
- Run a one-step Z-Image smoke config.
- Run Z-Image sampling-only configs.
- Compare sample behavior and logs with Musubi-backed Phase 1 execution.

**Exit criteria:**
- Two separate architectures run natively under one clean framework.
- Shared layers now reflect real commonality instead of guessed commonality.

---

### Phase 4: Performance and Reliability Hardening

**Intent:** Make the new framework credible for daily use.

**Runnable deliverable:**
- A “daily-driver” alpha for `Flux2` and `Z-Image` with stable smoke and short training runs.

**Core outputs:**
- performance benchmark scripts
- checkpoint resume tests
- parity dashboard or summary report
- documented known gaps

- [ ] Add performance baselines:
  - eager
  - compile
  - compile + prewarm
  - FP8 variants where supported
- [ ] Add resume / restore tests.
- [ ] Add output validation for sample generation.
- [ ] Add migration notes from Musubi-style flags to TOML config.
- [ ] Add known-regression gating so parity drift is visible early.

**How you can test it:**
- Run your normal short experiments in the new framework.
- Compare speed and sample quality against Musubi.
- Confirm that restart / resume behavior is trustworthy.

**Exit criteria:**
- The new framework is usable for repeated local experimentation.
- Performance-sensitive behaviors are measured, not assumed.

---

### Phase 5: Optional Interop Layer for PEFT and Diffusers

**Intent:** Borrow ecosystem standards without sacrificing runtime performance.

**Runnable deliverable:**
- Optional import/export and config translation tools.
- Not a full Diffusers-native execution path.

**Core outputs:**
- `northstar/interop/peft_adapter.py`
- `northstar/interop/diffusers_config_adapter.py`
- `northstar/interop/export_lora.py`
- `tests/interop/test_peft_roundtrip.py`

- [ ] Support PEFT-friendly LoRA import/export where possible.
- [ ] Support Diffusers-friendly scheduler and config translation where safe.
- [ ] Keep hot-path execution on the native runtime unless benchmarks prove otherwise.

**How you can test it:**
- Import a LoRA in one format and export it in another.
- Validate converted configs against smoke runs.

**Exit criteria:**
- Better interoperability without handing the critical runtime over to another stack.

---

### Phase 6: Cutover and Deprecation

**Intent:** Turn the extraction into a practical replacement for the targeted scope.

**Runnable deliverable:**
- A documented alpha or beta release with a clear “use this for Flux2 and Z-Image” story.

**Core outputs:**
- migration guide
- example configs
- release checklist
- remaining-gap tracker

- [ ] Publish tested example configs for both architectures.
- [ ] Provide a migration table:
  - old flag
  - new TOML field
  - notes
- [ ] Provide migration rules for:
  - old dataset TOML -> `dataset` section
  - old prompt file -> `sampling.prompts`
  - old cache scripts -> automatic cache preparation policy
- [ ] Mark unsupported legacy behaviors explicitly.
- [ ] Decide whether Musubi-backed mode remains as fallback or becomes deprecated.

**How you can test it:**
- Start a fresh experiment from TOML only.
- No shell-flag archaeology needed.

**Exit criteria:**
- The framework is no longer just an extraction experiment.
- It has a usable operator story.

## Suggested Validation Matrix

- `Flux2`
  - eager smoke
  - compile smoke
  - compile + prewarm smoke
  - compile + sampling LoRA
  - compile + FP8 + sampling LoRA
- `Z-Image`
  - eager smoke
  - compile smoke
  - compile + sampling LoRA
  - compile + FP8 if supported in the target path

## Time Estimate

- Phase 0-1:
  - `1-2 weeks`
- Phase 2:
  - `2-4 weeks`
- Phase 3:
  - `2-4 weeks`
- Phase 4:
  - `2-4 weeks`
- Phase 5-6:
  - `1-3 weeks`

**Total rough estimate for a good first version:** `2-4 months` for one strong engineer working steadily.

## Recommendation

- Do this as an extraction, not a “rewrite from first principles.”
- Make config, parity, and smoke deliverables the first-class milestones.
- Keep Musubi behavior as the reference until the new runtime proves itself.
- Treat PEFT and Diffusers as interoperability layers, not as the execution core, unless benchmarks later prove otherwise.
