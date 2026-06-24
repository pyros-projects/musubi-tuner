## Context

`ltx-musubi` already has local support for Krea2, Qwen Image, Z-Image, LTX2, sampling LoRA overlays, block-swap sampling overrides, Comfy LoRA export hooks, and dataset `caption_prefix`. The worktree is currently dirty from recent caption-prefix/gui changes, so this change must remain narrowly scoped and avoid unrelated cleanup.

Boogu-Image 0.1 is a 10B Lumina2-style flow-matching image model family. The initial target is Boogu Image Base text-to-image LoRA training. The model uses:

- a mixed double-stream/single-stream Boogu transformer;
- Qwen3-VL instruction features as text conditioning;
- a FLUX-compatible AutoencoderKL VAE released in the Boogu repo;
- native flow time where `t=0` is pure noise, `t=1` is clean, and the transformer predicts `clean - noise`;
- a Boogu time-shift schedule for sampling.

ai-toolkit added Boogu Image and Boogu Image Edit in commit `60c1ac6` under `extensions_built_in/diffusion_models/boogu_image/`. That implementation is a useful reference because it trims the upstream Boogu repo into training-relevant transformer, attention, RoPE, pipeline, and scheduler helpers. It also documents an important constraint: ai-toolkit trains from the clean bf16 Boogu repo and applies its own quantization; the HF `-fp8` sibling ships torchao float8 `.bin` weights and is not directly used for training there.

## Goals / Non-Goals

**Goals:**

- Add Boogu Image Base text-to-image LoRA training to `ltx-musubi`.
- Add Boogu latent and text-encoder cache scripts that match the repository's current cache conventions.
- Add Boogu snapshot sampling during training without reloading the text encoder for every sample.
- Add Boogu LoRA module targeting and checkpoint conversion tests before implementation.
- Add optional fp8-safe model loading/quantization compatible with trainable LoRA overlays.
- Add docs and `.pyro/boogu` convenience configs/scripts analogous to `.pyro/krea2`.
- Keep the implementation easy to compare against ai-toolkit and official Boogu source.

**Non-Goals:**

- Do not implement Boogu Image Edit/TI2I in the first change.
- Do not implement Boogu Turbo/DMD training or whole-model Turbo swap during snapshots.
- Do not port TeaCache, TaylorSeer, prompt rewriting, online demos, or inference-only accelerators.
- Do not merge ai-toolkit or upstream Boogu wholesale.
- Do not rewrite the LTX fork's trainer, dataset, or GUI architecture.
- Do not depend on the HF `Boogu-Image-0.1-Base-fp8` torchao `.bin` checkpoint as the initial training source.

## Decisions

### Decision: Implement Boogu Base T2I first

The first implementation will support `boogu_image` Base text-to-image LoRA training only.

Alternatives considered:

- Support Base, Edit, and Turbo together: rejected because Edit requires reference-image conditioning in both Qwen3-VL features and VAE latent streams, while Turbo changes sampling/training assumptions.
- Support only inference: rejected because the local goal is LoRA training and evaluation against Krea2.

### Decision: Use ai-toolkit's Boogu port as the main implementation reference

Port the training-relevant files from ai-toolkit's `boogu_image/src` and adapt them to Musubi's script/trainer layout. Use the official Boogu repository to cross-check model architecture, scheduler semantics, and VAE/text encoder expectations.

Alternatives considered:

- Start from official Boogu directly: rejected because ai-toolkit already removed training-irrelevant caches/fast paths and shows how to wire LoRA training.
- Import ai-toolkit code at runtime: rejected because `ltx-musubi` should remain self-contained.

### Decision: Follow the existing Krea2 port shape

Add isolated Boogu files and minimal shared glue:

- `src/musubi_tuner/boogu_image/*`
- `src/musubi_tuner/networks/lora_boogu_image.py`
- `src/musubi_tuner/boogu_image_cache_latents.py`
- `src/musubi_tuner/boogu_image_cache_text_encoder_outputs.py`
- `src/musubi_tuner/boogu_image_train_network.py`
- optional top-level wrappers if the branch still uses them.

Shared edits should be limited to architecture constants, cache save helpers, parser choices, and trainer hooks.

Alternatives considered:

- Fold Boogu into Qwen Image helpers: rejected because Boogu uses a different transformer, time convention, prompt format, and VAE.
- Create a new generic Lumina2 architecture layer first: rejected as premature for one model.

### Decision: Store natural-length instruction features and pad at call time

Boogu text caching will store one `(tokens, hidden)` instruction feature tensor per caption, plus enough metadata to identify the `boogu_image` embedding space. During train/sampling, batches pad those tensors to the maximum length and create a mask.

Alternatives considered:

- Store globally padded tensors: rejected because it wastes cache space and hides variable-length bugs.
- Re-encode prompts during every snapshot: rejected because Qwen3-VL is large and Krea2 already taught us to free the text encoder before training.

### Decision: Preserve Boogu's native time convention explicitly

Training glue must convert Musubi scheduler timesteps into Boogu native `t`, and model output must be negated where necessary so the shared loss target remains consistent. Snapshot sampling will use a small Boogu-native Euler loop with the time-shift schedule from ai-toolkit/Boogu.

Alternatives considered:

- Force Boogu through an existing Qwen/Krea scheduler path: rejected because Boogu's `t=0 noise, t=1 clean` convention and `clean - noise` prediction are inverted relative to common Musubi assumptions.

### Decision: Use bf16 weights plus local safe fp8 quantization

The initial training path should load bf16 `Boogu/Boogu-Image-0.1-Base`-style safetensors or local equivalents, then optionally apply the repository's existing scaled fp8/quantization path to selected transformer weights. Direct training from the HF `-fp8` torchao `.bin` release is out of scope until we know its serialization and trainable overlay behavior.

Alternatives considered:

- Train directly from `Boogu-Image-0.1-Base-fp8`: deferred because ai-toolkit explicitly avoids it and notes torchao/cache requirements.
- Cast every layer to fp8: rejected because norms, embeddings, modulation, and output projection are likely unsafe candidates.

### Decision: Add tests before implementation

The implementation should begin by adding failing CPU tests for:

- parser/import registration;
- Boogu architecture/cache metadata;
- instruction feature padding;
- time-shift schedule and output sign convention;
- LoRA target discovery;
- native/Musubi key conversion;
- post-save Comfy/native export behavior.

GPU/model-weight smoke tests remain manual because they require large weights and available VRAM.

Alternatives considered:

- Implement first, test after: rejected because this port touches model glue where silent key/sign/schedule mistakes are easy.

### Decision: Add Comfy/native LoRA conversion in the first pass

Saved Boogu LoRAs should keep the normal Musubi checkpoint by default and also emit a converted `.comfy.safetensors` or native `diffusion_model.*` checkpoint when conversion is enabled, matching the Krea2 workflow.

Alternatives considered:

- Leave conversion to a separate script only: rejected because Pyro's workflow expects immediate Comfy usability.
- Save only native keys: rejected because Musubi resume/load should continue using the normal LoRA format.

## Risks / Trade-offs

- Boogu's architecture is large and less battle-tested locally -> Start with CPU tests and one minimal GPU smoke command before longer runs.
- Output sign or timestep inversion can train the wrong target silently -> Add pure-tensor tests for the conversion and loss target.
- Qwen3-VL text encoding can exceed memory -> Cache text embeddings up front, release the encoder before training, and document low-VRAM options.
- fp8 trainability can be subtle with LoRA overlays -> keep bf16 as the baseline and make fp8 optional until a smoke run validates it.
- LoRA target names may not align with Comfy/native expectations -> write key-conversion tests from synthetic Boogu module names before saving real checkpoints.
- Edit/TI2I demand may arrive quickly -> keep file/module names extensible (`boogu_image` package, base trainer class) but do not wire edit behavior prematurely.

## Migration Plan

1. Add failing CPU tests for the Boogu support surface.
2. Port isolated Boogu model/helper files from ai-toolkit and official Boogu references.
3. Add dataset/cache architecture constants and save helpers.
4. Add Boogu cache scripts, trainer, LoRA network module, and snapshot sampler.
5. Add checkpoint conversion and post-save hooks.
6. Add docs and `.pyro/boogu` example configs/scripts.
7. Run the relevant CPU test subset.
8. Provide manual GPU smoke commands for cache, one-step train, snapshot sampling, bf16 baseline, and optional fp8 mode.

Rollback is file-level: remove the Boogu package/wrappers/tests/docs/examples and revert the small shared architecture/trainer registrations. The change should not rewrite existing Krea2/Qwen/LTX paths.

## Open Questions

- Which local Boogu model files should the `.pyro/boogu` config prefer once Pyro downloads/converts them: HF repo paths, `/home/pyro/models/comfy`, or a dedicated `/home/pyro/models/boogu` layout?
- Should the first fp8 path use Musubi's scaled fp8 storage pattern or a narrower "selected Linear only" cast list derived from ai-toolkit/Boogu module names?
- Do current Comfy Boogu LoRA loaders expect `diffusion_model.*` keys exactly, or a slightly different prefix for this model?
- Should `boogu_image_edit` be a separate follow-up OpenSpec change or a second phase inside this one after Base passes smoke tests?
