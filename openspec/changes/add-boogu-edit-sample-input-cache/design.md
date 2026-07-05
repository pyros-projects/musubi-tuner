## Context

Boogu training snapshot sampling currently pre-encodes text prompts with Qwen3-VL in `process_sample_prompts()`, releases the text encoder, then lets the shared trainer load the VAE and transformer. That lifecycle is useful because the expensive text encoder does not overlap the train loop.

Boogu Edit needs two kinds of source-image conditioning during sampling:

- Qwen3-VL vision tokens in the positive instruction path.
- VAE reference latents passed to `ref_image_hidden_states` for both conditional and unconditional denoise calls.

The local ComfyUI Boogu node follows that same split. The current Musubi Boogu sampler already passes `ref_image_hidden_states=None`; the native transformer/pipeline path supports non-`None` reference latents. The missing piece is prompt-level image resolution, pre-encoding, and sample-time wiring.

The shared prompt TOML loader already merges `[prompt]` defaults into each `[[prompt.subset]]` entry and lets subset keys override root keys. This makes global and per-prompt input images possible without changing the shared prompt loader.

## Goals / Non-Goals

**Goals:**

- Let Boogu sample prompt TOML declare one optional edit input image globally or per prompt.
- Pre-cache all Boogu sample input image data before the transformer is loaded.
- Match Boogu Edit conditioning semantics: image-aware positive instruction features, image-free negative instruction features, and reference latents on both conditional branches.
- Keep existing no-image Boogu snapshot sampling behavior unchanged.
- Fail early for missing or ambiguous sample input image configuration.

**Non-Goals:**

- Changing the training batch contract. Training remains output-only and continues using cached dataset latents/text features.
- Teaching prior-post edit concepts from paired input/output training data.
- Supporting multiple input images per Boogu sample prompt in this change.
- Reworking the shared trainer lifecycle or the shared prompt TOML resolver.
- Changing final VAE decode/offload behavior during sampling.

## Decisions

1. Use `input_image` as the canonical prompt key.

   Boogu prompt TOML will use `input_image` for a single edit source image. Because the shared prompt loader already handles root defaults and subset overrides, this key can be placed under `[prompt]` or under an individual `[[prompt.subset]]`.

   Alternative considered: reuse `image_path`. That name is already broad across video/image sampling code and is easier to confuse with output or dataset image paths.

   Alternative considered: only reuse Flux/Klein `control_image_path`. That improves cross-model familiarity but makes the Boogu edit source sound like ControlNet-style conditioning. The implementation can still accept `control_image_path` as a compatibility alias when it is a string or single-item list.

2. Pre-cache image conditioning in Boogu `process_sample_prompts()`.

   `process_sample_prompts()` will continue to be the Boogu sampling preparation hook. It will encode Qwen3-VL text/image prompt features, release the text encoder, then temporarily load the Boogu VAE only when at least one sample input image is present. The VAE will encode reference latents, store them on CPU in the sample parameter records, and be released before the shared trainer proceeds to its normal VAE and transformer loading.

   Alternative considered: encode reference images inside `do_inference()`. That is simpler locally, but it overlaps VAE encode work with transformer sampling and undermines the low-memory sampling path.

   Alternative considered: change the shared trainer to pass the normal sampling VAE into `process_sample_prompts()`. That would avoid loading the VAE twice, but it broadens a Boogu-specific change into shared lifecycle code.

3. Match Comfy/reference preprocessing sizes.

   Qwen3-VL image inputs should be capped around the reference 384x384 area before processor encoding. VAE reference latents should be resized around the reference 1024x1024 area and aligned to 16 pixels before VAE encode. This mirrors the local Boogu Comfy node and keeps patching/latent shape behavior compatible with the transformer.

   Alternative considered: feed original image sizes into both paths. That risks excessive VLM memory and can produce latent sizes not aligned to Boogu patching assumptions.

4. Cache keys must include conditioning mode and image identity.

   Text-only positive prompts can keep prompt-string cache behavior. Image-conditioned positive prompts need a key that includes the resolved input image path, because the same edit instruction against two different images must not reuse the same Qwen3-VL features. Reference latent cache keys can be keyed by resolved input image path.

   Alternative considered: disable prompt feature reuse when images are present. That is correct but wastes work for repeated prompts/images in snapshot files.

5. Keep sample-time data small and explicit.

   Each sample parameter will carry optional CPU reference latents, likely as `boogu_ref_image_hidden_states` or equivalent. `do_inference()` will move them to the accelerator device and DiT dtype immediately before denoising and pass the same nested batch list to conditional and unconditional model calls. No-image samples keep passing `None`.

   Alternative considered: preserve path strings and lazy-load images during sampling. That avoids CPU tensor storage but moves IO and VAE work into the most memory-sensitive stage.

## Risks / Trade-offs

- VAE loads twice for runs with Boogu edit sample images -> keep this change Boogu-local and optimize later only if startup cost becomes material.
- Reference preprocessing may drift from official Boogu if upstream changes its constants -> keep the 384-area VLM and 1024-area latent values close to the local Comfy implementation and name them clearly.
- `control_image_path` compatibility could hide multi-image configs -> accept only a string or single-item list and raise a clear error for multiple paths.
- Image-conditioned prompt caching can collide if keyed only by prompt text -> include resolved image path and conditioning mode in the cache key.
- CPU cached reference latents add memory proportional to sample image count -> cache one latent per unique image path and share it across sample parameters.

## Migration Plan

This is an additive sampling feature. Existing Boogu prompt files without `input_image` continue to run as text-to-image samples. New prompt files can add `input_image` under `[prompt]` for a global edit source or under `[[prompt.subset]]` for a per-prompt override.

Rollback is removing the new prompt keys or reverting the change; no dataset cache migration is required.

## Open Questions

- Should `image_path` be accepted as a deprecated alias, or is `input_image` plus `control_image_path` enough?
- Should later work expose multiple reference images once the single-image path is validated against Boogu Edit training previews?
