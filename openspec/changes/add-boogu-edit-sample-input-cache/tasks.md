## 1. Prompt Config and Validation

- [x] 1.1 Add a Boogu helper that resolves one sample edit input image from `input_image` or compatible `control_image_path` values after prompt root/subset inheritance.
- [x] 1.2 Normalize resolved sample input image paths, validate they exist before training, and keep no-image prompts valid.
- [x] 1.3 Reject unsupported multiple-image values with a clear prompt-processing error.
- [x] 1.4 Add tests for global `input_image`, per-prompt override, `control_image_path` string and single-item list aliases, no-image prompts, missing files, and multiple-image rejection.

## 2. Qwen3-VL Conditioning Cache

- [x] 2.1 Add Boogu edit/drop message or processor-input helpers for Qwen3-VL image-conditioned positive prompts while preserving the existing text-to-image prompt path.
- [x] 2.2 Extend `process_sample_prompts()` to cache text-only and image-conditioned instruction features with cache keys that include prompt text, conditioning mode, and resolved image path when present.
- [x] 2.3 Store cached instruction features on CPU in bf16-compatible tensors and keep negative prompt features image-free.
- [x] 2.4 Add tests proving repeated prompt/image pairs reuse cached features while identical prompt text with different images does not collide.

## 3. Reference Latent Cache

- [x] 3.1 Add Boogu sample-image preprocessing for VAE reference latents using the reference 1024x1024-area resize and 16-pixel alignment.
- [x] 3.2 Temporarily load the Boogu VAE inside `process_sample_prompts()` only when sample input images exist.
- [x] 3.3 Encode one CPU reference latent per unique resolved sample image path and release the temporary VAE before returning sample parameters.
- [x] 3.4 Store optional cached reference latents in each relevant sample parameter in a shape that can be converted to the transformer's `List[List[Tensor]]` `ref_image_hidden_states` input.
- [x] 3.5 Add tests or mocks verifying no-image prompt files do not load the temporary VAE and image prompt files reuse cached latents.

## 4. Snapshot Sampling Wiring

- [x] 4.1 Update Boogu `do_inference()` to move cached reference latents to the accelerator device and DiT dtype immediately before denoising.
- [x] 4.2 Pass cached reference latents as `ref_image_hidden_states` to both conditional and unconditional denoise calls when a sample has an edit input image.
- [x] 4.3 Preserve existing text-only CFG and no-CFG behavior by continuing to pass `None` for samples without edit input images.
- [x] 4.4 Add focused sampler tests proving reference latents are provided to both CFG branches and omitted for text-only samples.

## 5. Verification and Docs

- [x] 5.1 Document the Boogu sample prompt `input_image` key and `control_image_path` compatibility alias in the nearest Boogu sampling docs or example prompt surface.
- [x] 5.2 Run the focused Boogu sampling/prompt tests added for this change.
- [x] 5.3 Run a lint or import check for the touched Boogu modules and tests.
- [x] 5.4 Run `openspec validate add-boogu-edit-sample-input-cache --strict`.
- [x] 5.5 If GPU resources are available, run a short Boogu Edit snapshot smoke sample with one `input_image` prompt and verify an output image is written. Deferred because the GPU is currently in use.
