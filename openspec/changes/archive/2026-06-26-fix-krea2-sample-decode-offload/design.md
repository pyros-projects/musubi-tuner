## Context

Krea2 sampling currently performs denoising and VAE decode inside Krea-specific inference helpers. In the training snapshot path, `hv_train_network.sample_images()` can move the transformer to CPU after `sample_image_inference()` returns when `--sample_with_offloading` is enabled, but Krea2 has already moved the Qwen-Image VAE onto the accelerator and decoded by then. This means decode can peak with both the Krea2 transformer and VAE resident on GPU.

LTX2 already avoids this shape by explicitly offloading the transformer before final VAE decode. Krea2 should use the same staged memory pattern while preserving existing behavior for users who do not request sampling offload.

## Goals / Non-Goals

**Goals:**

- Offload the Krea2 transformer after denoising and before VAE decode when sampling offload is requested.
- Keep the default resident-transformer decode path unchanged when sampling offload is not requested.
- Preserve training snapshot behavior across global sampling LoRA, prompt-local LoRA, and block-swap sampling overrides.
- Add a standalone Krea2 generation flag with matching semantics for manual smoke testing.
- Cover the behavior with CPU fake-object tests that assert call ordering and restoration without model weights.

**Non-Goals:**

- Changing Krea2 denoising math, timestep schedules, prompt encoding, or VAE decode quality.
- Making sampling faster. This change trades some latency for lower decode-time VRAM.
- Reworking the shared Musubi sampling loop.
- Running real GPU smoke tests while the GPU is busy.

## Decisions

### Add a Krea2 decode-offload helper

Add a small Krea2-specific helper/context manager that can move the transformer off the accelerator before VAE decode and restore the expected state after decode.

Rationale: Krea2 owns the decode point, so the offload hook needs to live inside the Krea2 inference path. Relying only on the shared trainer callback is too late because the callback fires after Krea2 decode has already returned.

Alternative considered: split Krea2 `sample_image_inference()` into denoise and decode phases and make the shared trainer own decode offload. That is larger and riskier because other model trainers use the shared sampling surface differently.

### Reuse `--sample_with_offloading` for training snapshots

For in-training snapshots, decode-time offload should be enabled by the existing `--sample_with_offloading` flag.

Rationale: operators already use that flag to request lower-VRAM sampling. Making Krea2 honor it before decode matches LTX2's intent and avoids another training-only switch.

Alternative considered: add a new training flag such as `--sample_offload_before_decode`. That would be more explicit but easier to miss and inconsistent with the existing operator mental model.

### Add a standalone Krea2 offload flag

Add a standalone generation flag, preferably `--sample_with_offloading`, to `krea2_generate_image.py` and pass it into the Krea2 sampling/decode path.

Rationale: LTX standalone generation already uses this naming, and Pyro can use the standalone script as a lower-risk smoke surface before long training runs.

Alternative considered: `--offload_transformer_for_decode`. The name is precise, but it diverges from established Musubi sampling vocabulary.

### Preserve block-swap-aware movement

When the transformer has block-swap helpers, the decode-offload helper should use the same move/prepare calls as the Krea2 sampling block-swap path rather than only calling `.to("cpu")` / `.to(device)`.

Rationale: Krea2 sampling may be running with `--blocks_to_swap` or `--sample_blocks_to_swap`. The helper must not leave swapped blocks in a state that breaks the next prompt or training restore.

Alternative considered: disable decode offload whenever block swap is active. That would avoid complexity but would remove the feature from the exact low-VRAM setups that need it most.

## Risks / Trade-offs

- Transformer offload before decode adds latency -> Keep it opt-in and no-op by default.
- Block-swap restore can be easy to get subtly wrong -> Add fake-object tests that assert the sequence of offload, VAE move/decode, and restore calls.
- VAE decode exceptions could leave the transformer offloaded -> Use `try/finally` so the state after failure is compatible with the outer training sampler cleanup.
- CUDA cache cleanup can hide real leaks if overused -> Limit cache cleanup to the explicit offload boundary.

## Migration Plan

1. Add CPU tests for Krea2 decode-offload ordering and standalone flag parsing.
2. Implement the Krea2 helper and wire it into training snapshot decode.
3. Add and wire the standalone Krea2 generation flag.
4. Run CPU tests and help output checks.
5. Defer real GPU smoke until the current Krea2 experiment is done.

Rollback is deleting the helper wiring and standalone flag; default behavior remains unchanged when the flag is absent, so the blast radius is small.

## Open Questions

- During training snapshots, should the helper restore the transformer to GPU immediately after decode, or leave it CPU/offloaded and let the existing shared sampling loop restore it before the next prompt? The implementation should prefer the path that avoids redundant movement while still passing fake-object call-order tests.
