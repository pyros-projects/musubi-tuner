## Context

Upstream Musubi has a Krea 2 implementation in `/tmp/krea2-fp8-spelunk/musubi-tuner` around commit/tag `v0.3.4`. The LTX sandbox at `/home/pyro/repos/ltx-musubi/musubi-tuner` is a divergent fork on branch `ltx-2` with local dirty work in trainer, dataset, prompt-local sampling LoRA, compile prewarm, and LTX2 tests. A git merge or broad upstream transplant is high risk.

Krea 2 is a text-to-image model with a single-stream MMDiT, Qwen3-VL-4B text encoder, and Qwen-Image VAE. Upstream Krea 2 uses newer split modules such as `dataset/architectures.py`, `dataset/cache_io.py`, `training/parser_common.py`, and `training/trainer_base.py`; LTX keeps those concerns in `dataset/image_video_dataset.py` and `hv_train_network.py`. The port must adapt upstream Krea 2 code to the LTX layout.

The target workflow is Pyro's RAW/base training plus Turbo-LoRA snapshot preview:

- train Krea 2 RAW/base normally;
- keep the trainable LoRA applied by Musubi's normal training network;
- temporarily apply the frozen Turbo LoRA only during snapshot sampling;
- support scaled fp8 storage for the large Krea 2 transformer.

## Goals / Non-Goals

**Goals:**

- Add Krea 2 RAW/base LoRA training to LTX Musubi.
- Add Krea 2 latent and text-encoder cache scripts using Qwen-Image VAE and Qwen3-VL embeddings.
- Add Krea 2 snapshot sampling during training, including CFG-capable RAW sampling and Turbo-LoRA fast preview sampling.
- Add Krea 2 scaled-fp8 loading compatible with training, sampling LoRA overlays, block swap where supported, and torch compile where the existing infrastructure allows it.
- Preserve existing LTX fork behavior and dirty local changes.
- Keep the implementation easy to compare against upstream by copying isolated Krea 2 files where possible and hand-porting only shared glue.

**Non-Goals:**

- Do not port upstream's full trainer/dataset module split into LTX.
- Do not make a direct git merge from upstream Musubi.
- Do not implement trainer-student de-turborisation, teacher losses, or other optimization experiments.
- Do not make upstream `--turbo_dit` whole-base swapping the primary snapshot preview path.
- Do not support Krea 2 image-to-image, edit, control, video, or layered datasets in this change.

## Decisions

### Decision: Manual port over git merge

Port upstream Krea 2 files manually and adapt integration points to the LTX fork.

Alternatives considered:

- Direct merge/cherry-pick from upstream: rejected because LTX is a fork with substantial divergence and local dirty changes.
- Rebase LTX onto upstream: rejected because it risks the LTX2 experimentation surface and unrelated local work.

### Decision: Keep upstream Krea 2 code isolated where possible

Copy upstream files such as `src/musubi_tuner/krea2/*`, `src/musubi_tuner/networks/lora_krea2.py`, Krea2 cache scripts, generator, docs, and CPU tests with minimal edits. Adapt imports and shared glue to LTX only where required.

Alternatives considered:

- Rewrite Krea 2 against LTX from scratch: rejected because upstream has working architecture details, text packing, GQA handling, and fp8 scope.
- Import upstream split modules wholesale: rejected because it would create duplicate architecture/cache/trainer abstractions inside LTX.

### Decision: Use sampling LoRA adapter instead of upstream Turbo-DiT swap

Use LTX's existing `--sampling_lora_weight` / `--sampling_lora_multiplier` path for Turbo preview. The train loop trains RAW/base normally; snapshot sampling temporarily applies the frozen Turbo LoRA adapter and restores training state afterward.

Alternatives considered:

- Port upstream `--turbo_dit` full-base swap: rejected as initial scope because it conflicts with block swap, duplicates a workflow Pyro does not want, and swaps the whole base instead of applying the Turbo LoRA adapter.
- Merge Turbo LoRA into the base at load time: rejected for training because it would train on Turbo-adapted weights instead of RAW/base.

### Decision: Normalize local Turbo LoRA keys in the Krea2 trainer

Add a `normalize_sampling_lora_weights()` override for Krea 2, analogous to the Qwen Image and LTX2 hooks. It must convert local `diffusion_model.<path>.lora_*` keys into Musubi `lora_unet_<path_with_underscores>.lora_*` keys accepted by `networks.lora_krea2`.

Alternatives considered:

- Require Pyro to convert the Turbo LoRA externally: rejected because the local file already works in diffusion-pipe and this should be robust in the training config.
- Add generic native-key normalization to the base trainer: rejected for now because Krea2/Qwen/LTX key layouts differ and model-specific hooks are already the local pattern.

### Decision: Add explicit Krea 2 sample `mu` support

Krea 2 RAW inference uses a resolution-aware shift; Turbo preview should use fixed `mu=1.15`. Since upstream training sampler pins fixed `mu` only when `--turbo_dit` is set, the LTX port must support a Krea2-specific sample override, likely via prompt `--mu 1.15` and/or a model-specific default when a sampling LoRA is active.

Alternatives considered:

- Rely on `--fs` discrete flow shift: rejected because upstream Krea2 sampler uses `mu`, not the generic FlowMatch scheduler path.
- Always force `mu=1.15`: rejected because RAW CFG snapshots should still be able to use the resolution-aware schedule.

### Decision: Require scaled fp8 for Krea 2 fp8

Krea 2 `--fp8_base` must require `--fp8_scaled`; plain full-model fp8 casting is rejected because norms and modulation must not be cast to fp8. The target key set is `blocks.`, with excludes `mod.`, `norm`, and `txtfusion`.

Alternatives considered:

- Support plain `--fp8_base`: rejected because upstream explicitly warns this breaks Krea 2.
- Port upstream fp8 utility wholesale: rejected because LTX has local dtype-handling changes in `fp8_optimization_utils.py`; Krea2 does not need upstream's prequantized-fp8 knob for the bf16 RAW plus bf16 Turbo-LoRA workflow.

### Decision: Bump Transformers rather than vendor Qwen3-VL

Use upstream's dependency level for Qwen3-VL support, most likely `transformers==4.57.6`. The current LTX environment is `transformers==4.56.1`, which lacks `Qwen3VLConfig` and `Qwen3VLForConditionalGeneration`.

Alternatives considered:

- Vendor upstream `hidream_o1/qwen3_vl_transformers.py`: rejected as initial scope because it pulls in a large generated Transformers model and still depends on Qwen3-VL config classes.
- Keep the old dependency and fail at runtime: rejected because text encoder caching and snapshot prompt encoding are core requirements.

## Risks / Trade-offs

- Krea 2 text encoder memory is large -> Mitigate by matching upstream: cache training text embeddings, encode snapshot prompts up front, then delete/offload the encoder and clear CUDA memory.
- Dependency bump could affect other models -> Mitigate with CPU import tests for existing Qwen/Z-Image/LTX surfaces and keep the bump narrow.
- Sampling LoRA overlays on fp8 weights can be subtle -> Mitigate by reusing LTX's existing runtime overlay fallback for float8 modules and adding key-normalization tests.
- Upstream Krea2 assumes the newer split layout -> Mitigate by adapting only the required architecture/cache/trainer hooks into LTX's current files.
- Krea 2 GQA can break shared attention backends -> Mitigate by porting upstream's GQA head expansion and non-contiguous reshape fixes in `modules/attention.py`.
- Whole-DiT Turbo swap remains unsupported -> Mitigate by documenting that the supported preview path is Turbo LoRA as sampling adapter.

## Migration Plan

1. Initialize Krea 2 files and wrappers in LTX without touching unrelated LTX2 behavior.
2. Add shared dataset/trainer/attention/dependency glue.
3. Add Krea2-specific sampling-LoRA normalization and sample `mu` handling.
4. Add docs and CPU tests.
5. Verify imports, parser behavior, timestep behavior, prompt text compaction, and LoRA key normalization locally.
6. Leave GPU smoke tests to Pyro when the GPU is available.

Rollback is file-level: remove the new Krea2 files and revert the small shared glue patches. The change should avoid structural rewrites so rollback stays simple.

## Open Questions

- Should Turbo-LoRA snapshot sampling automatically use `mu=1.15` whenever a Krea2 sampling LoRA is active, or should it require explicit `--mu 1.15` in sample prompts/configs?
- Should upstream `--turbo_dit` be preserved as an explicitly unsupported parser option with a clear error, or simply omitted from the LTX Krea2 trainer?
- Should Krea2 docs include Pyro's local model paths as comments/examples, or stay upstream-generic?
