## Context

Boogu snapshot sampling is implemented in `boogu_image_train_network.py` using a native Euler denoise loop over `boogu_time_schedule(...)`. The current loop has no progress bar, so a slow sample appears idle during training.

The shared trainer already supports `--sample_with_offloading` around prompt-level sampling: it can move the transformer to CPU between prompts and move it back before each prompt. That outer offload does not help VAE decode in Boogu because `do_inference()` decodes latents before returning to the shared trainer. At that point the transformer is still on the sampling device, so `vae.to(device)` can stack VAE memory on top of the transformer.

Krea2 has a focused helper for this exact issue: wait for pending block-swap transfers, move the transformer to CPU, clean the device, run VAE decode, and optionally restore the transformer.

## Goals / Non-Goals

**Goals:**

- Show Boogu sampling denoise progress during training snapshots.
- Ensure `--sample_with_offloading` moves the Boogu transformer off the sampling device before VAE decode.
- Preserve current device behavior when sample offloading is disabled.
- Keep decode cleanup and restore behavior safe if VAE decode raises.
- Cover the behavior with CPU tests that do not require large weights or GPU smoke.

**Non-Goals:**

- Do not implement real Boogu block swap in this change.
- Do not change Boogu sampling schedule, CFG behavior, prompt caching, or output quality.
- Do not add a new CLI flag.
- Do not require real Boogu weights in automated tests.

## Decisions

### Decision: Use the same decode-offload contract as Krea2

Boogu should call a helper around VAE decode when `args.sample_with_offloading` is true. The helper must:

- call `_wait_for_pending_block_swaps()` on the transformer when available;
- move the transformer to CPU before `vae.to(device)`;
- clean the sampling device before decode;
- leave prompt-level restoration to the shared sampler by default;
- still let the VAE cleanup path run if decode fails.

Alternatives considered:

- Rely only on shared prompt-level offload: rejected because VAE decode happens inside `do_inference()`, before prompt-level offload regains control.
- Inline a Boogu-only helper: acceptable if extraction becomes noisy, but the preferred route is a shared helper because Krea2 already has tests for the same lifecycle.

### Decision: Keep restoration owned by the outer sampler

When `--sample_with_offloading` is enabled, Boogu decode offload should use `restore_after=False`, matching the Krea2 training-sampler path. The shared sampler already moves the transformer back to the sampling device before the next prompt.

Alternatives considered:

- Restore immediately after VAE decode: rejected for training snapshots because it needlessly reloads the transformer before the shared sampler will move it back to CPU between prompts.

### Decision: Add progress at the denoise-loop boundary

The progress bar should wrap the native Boogu timestep transition loop and report `total=len(times) - 1` with a concise description such as `Denoising steps`. This makes the progress count match the configured sample step count after schedule construction.

Alternatives considered:

- Add logging before and after sampling only: rejected because it does not reveal slow or stuck denoise progress.

## Risks / Trade-offs

- Shared helper extraction could accidentally regress Krea2 -> keep existing Krea2 decode-offload tests in the verification set.
- Progress bars can become noisy under distributed launch -> use the repository's existing `tqdm` style and keep the description short.
- Decode failure could leave VAE or transformer on the wrong device -> test failure paths with fake modules and make cleanup happen in `finally`.
- Future real block-swap support needs synchronization before decode -> helper must continue to call `_wait_for_pending_block_swaps()` when present.

## Migration Plan

1. Add failing CPU tests for Boogu decode-offload ordering and disabled-offload behavior.
2. Add a failing CPU test that Boogu's denoise loop wraps its timestep transitions in `tqdm` with the expected total.
3. Extract or reuse the Krea2 decode-offload helper and update imports if needed.
4. Wrap Boogu VAE decode with the helper.
5. Add the Boogu denoise progress bar.
6. Run focused CPU tests for Boogu and Krea2 decode offload.

Rollback is file-level: remove the Boogu calls to the helper/progress wrapper and, if extracted, restore the Krea2-local helper.

## Open Questions

- None for proposal scope. Real GPU sampling should be deferred until the GPU is free, but the CPU tests can validate ordering and progress wiring now.
