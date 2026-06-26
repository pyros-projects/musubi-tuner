## Context

`ltx-musubi` already has local support for Krea2, Qwen Image, Z-Image, LTX2, sampling LoRA overlays, block-swap sampling overrides, Comfy LoRA export hooks, and dataset `caption_prefix`. The worktree was clean on branch `ltx-2` before this reference refresh on 2026-06-26, so this change should remain narrowly scoped and avoid unrelated cleanup.

Boogu-Image 0.1 is a 10B Lumina2-style flow-matching image model family. The initial target is Boogu Image Base text-to-image LoRA training. The model uses:

- a mixed double-stream/single-stream Boogu transformer;
- Qwen3-VL instruction features as text conditioning;
- a FLUX-compatible AutoencoderKL VAE released in the Boogu repo;
- native flow time where `t=0` is pure noise, `t=1` is clean, and the transformer predicts `clean - noise`;
- a Boogu time-shift schedule for sampling.

ai-toolkit added Boogu Image and Boogu Image Edit in commit `60c1ac6` under `extensions_built_in/diffusion_models/boogu_image/`. On 2026-06-26, `https://github.com/ostris/ai-toolkit` was cloned into a temp directory; current `main` resolved to commit `4a99ddabadbb27e5471d7023c9b429b5e0b39cb6` (`Fix breaking change with diffusers qwen image`). The Boogu directory had no diffs between `60c1ac6` and `4a99ddabadbb27e5471d7023c9b429b5e0b39cb6`, but the current commit is still the pinned reference so adjacent toolkit, network, quantization, and training fixes remain visible during the port.

The relevant ai-toolkit files at that pinned commit are:

- `extensions_built_in/diffusion_models/boogu_image/__init__.py`
- `extensions_built_in/diffusion_models/boogu_image/boogu_image.py`
- `extensions_built_in/diffusion_models/boogu_image/boogu_image_edit.py`
- `extensions_built_in/diffusion_models/boogu_image/src/attention_processor.py`
- `extensions_built_in/diffusion_models/boogu_image/src/block_lumina2.py`
- `extensions_built_in/diffusion_models/boogu_image/src/embeddings.py`
- `extensions_built_in/diffusion_models/boogu_image/src/pipeline.py`
- `extensions_built_in/diffusion_models/boogu_image/src/rope.py`
- `extensions_built_in/diffusion_models/boogu_image/src/transformer.py`

That implementation is a useful reference because it trims the upstream Boogu repo into training-relevant transformer, attention, RoPE, pipeline, and scheduler helpers. It also documents an important constraint: ai-toolkit trains from the clean bf16 Boogu repo and applies its own quantization; the HF `-fp8` sibling ships torchao float8 `.bin` weights and is not directly used for training there.

Official source inspection on 2026-06-26:

- Official repository: `https://github.com/boogu-project/Boogu-Image`, cloned at commit `13853b77b6ba8dd393685231b662acf9f90730e9` (`misc: update inference scripts default res`).
- Official Base model card/repo: `https://huggingface.co/Boogu/Boogu-Image-0.1-Base`, HF model SHA `e10ed9d3691f25416d439aeb4de33b22bf9039c1`.
- The Base HF repo is a Diffusers `BooguImagePipeline` with subfolders `mllm`, `processor`, `scheduler`, `transformer`, and `vae`.
- Text encoder classes are `Qwen3VLForConditionalGeneration` / `Qwen3VLProcessor`; the HF `mllm/config.json` reports `model_type = qwen3_vl`, text hidden size `4096`, and 36 text layers.
- VAE class is Diffusers `AutoencoderKL`; the HF VAE config uses `latent_channels = 16`, `scaling_factor = 0.3611`, `shift_factor = 0.1159`, `sample_size = 1024`, and is described upstream as FLUX.1 VAE based.
- Scheduler class is `FlowMatchEulerDiscreteScheduler`; the HF scheduler config uses `do_shift = true`, `dynamic_time_shift = false`, `time_shift_version = v1`, `seq_len = 4096`, and `num_train_timesteps = 1000`.
- Transformer config uses `hidden_size = 3360`, `num_layers = 40`, `num_double_stream_layers = 8`, `num_refiner_layers = 2`, `num_attention_heads = 28`, `num_kv_heads = 7`, `patch_size = 2`, `in_channels = 16`, `instruction_feat_dim = 4096`, and `timestep_scale = 1000.0`.
- Supported first-change scope remains Base text-to-image LoRA training. Official Base supports T2I at 1K/1.5K/2K with 25-50 steps and guidance around 2.0-5.0; Edit/TI2I and Turbo/DMD remain out of scope for this change.
- Local model metadata check found `/home/pyro/models/comfy/diffusion_models/boogu_image_base_bf16.safetensors`, `/home/pyro/models/comfy/diffusion_models/boogu_image_edit_bf16.safetensors`, `/home/pyro/models/comfy/diffusion_models/boogu_image_edit_fp8_scaled.safetensors`, `/home/pyro/models/comfy/loras/boogu/boogu_image_turbo_lora_rank_128_bf16.safetensors`, `/home/pyro/models/comfy/vae/ae.safetensors`, `/home/pyro/models/comfy/vae/flux1_vae_bf16.safetensors`, `/home/pyro/models/comfy/text_encoders/qwen3vl_8b_fp8_scaled.safetensors`, and `/home/pyro/models/qwen3-vl-4b` for processor/tokenizer assets.
- GPU/model-load smoke was later validated by the local `lucy-adamw8bit` run on 2026-06-26. That run used the Comfy Boogu Base transformer, FLUX/Boogu VAE, Comfy `qwen3vl_8b_fp8_scaled.safetensors` text encoder with `/home/pyro/models/qwen3-vl-4b` processor assets, and the Boogu turbo LoRA for snapshot sampling. It completed training, snapshot sampling, checkpoint save, state save, and Comfy-format LoRA export under `/home/pyro/models/_out/boogu/lucy-adamw8bit`.

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

Port the training-relevant files from ai-toolkit's `boogu_image/src` at pinned commit `4a99ddabadbb27e5471d7023c9b429b5e0b39cb6` and adapt them to Musubi's script/trainer layout. Use ai-toolkit's current adjacent trainer/network/quantization code at the same commit for implementation clues, and use the official Boogu repository to cross-check model architecture, scheduler semantics, and VAE/text encoder expectations.

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

## Resolved Questions And Follow-ups

- Local Boogu examples prefer the existing `/home/pyro/models/comfy` layout plus `/home/pyro/models/qwen3-vl-4b` processor assets, because the successful `lucy-adamw8bit` smoke used those paths directly.
- The first fp8 path uses Musubi's scaled fp8 loading/monkey-patch pattern with Boogu-specific target/exclude lists. The `lucy-adamw8bit` smoke validated fp8 base loading together with trainable LoRA overlays and sampling LoRA runtime overlays.
- Converted LoRA checkpoints use native Boogu `diffusion_model.*` keys. The smoke produced `.comfy.safetensors` exports with the expected metadata and tensor layout; a live ComfyUI import/render can be a follow-up validation if needed, but is not required for this OpenSpec change.
- Boogu Image Edit/TI2I and Turbo/DMD training remain separate follow-up work outside this Base T2I LoRA change.
