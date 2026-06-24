## Context

Krea2 block swap is currently configured once from `--blocks_to_swap` and then reused for both training and in-training sample generation. `sample_images()` switches the model's offloader into forward-only mode for sampling, but the same number of blocks remain swapped. This is correct for memory safety, yet it can make sampling much slower because each denoise step repeatedly transfers Krea2 main blocks between CPU and GPU.

Sampling runs under `torch.no_grad()` and normally has lower activation pressure than training. In the common Krea2 LoRA workflow, the user may need block swap to make train-time batches fit but can afford fewer swapped blocks, or no swapped blocks, while generating snapshots.

## Goals / Non-Goals

**Goals:**

- Add a Krea2-compatible `--sample_blocks_to_swap` option.
- Default to the existing behavior when the option is unset.
- Allow `--sample_blocks_to_swap 0` to run snapshot denoising with all Krea2 blocks on the accelerator.
- Allow a positive value to use fewer or otherwise explicit swapped blocks during sampling.
- Restore the original training block-swap count and device layout after sampling, including after exceptions.
- Keep the override CPU-testable without requiring real Krea2 weights or CUDA.

**Non-Goals:**

- Do not change the meaning of train-time `--blocks_to_swap`.
- Do not create a second transformer instance for sampling.
- Do not make block swap dynamically adapt per prompt or after OOM.
- Do not solve generic block-swap overrides for every architecture unless the shared trainer hook can do so without changing other model contracts.

## Decisions

1. Add a sampling-specific CLI option in the shared trainer parser.

   `--sample_blocks_to_swap` defaults to `None`, meaning "inherit `--blocks_to_swap`". Values are integers. `0` disables block swap during snapshot sampling. Positive values request that number of swapped blocks during sampling.

   Alternative considered: reusing `--sample_with_offloading`. That flag already means "move transformer to CPU between prompts", not "change per-forward block streaming", so overloading it would make configs confusing.

2. Implement the actual state transition in the Krea2 transformer.

   Krea2 should expose a small helper/context manager that temporarily changes block-swap count, waits for pending transfers, updates both the model-level `blocks_to_swap` and the offloader's count, moves block weights into the requested device layout, and restores the original state in `finally`.

   Alternative considered: mutate `transformer.blocks_to_swap` directly in `hv_train_network.sample_images()`. That is too fragile because the offloader also stores its own swap count and may have pending asynchronous transfers.

3. Keep `None` as no-op and make `0` explicit.

   This preserves existing configs exactly while making the fast sampling mode opt-in. A zero value is useful and must not be treated like false/missing.

4. Validate impossible override requests early.

   Krea2 can only swap up to `len(blocks) - 2`. The helper should reject values outside the valid range with a clear error. If a positive sampling override is requested while block swap was not initialized for training, the implementation should either initialize a forward-only offloader safely or fail clearly. For the first implementation, failing clearly is acceptable because the target use case is reducing train-time swap during sampling.

5. Restore before returning to training.

   The trainer already calls `switch_block_swap_for_training()` in the sampling `finally` path. The override restoration should run before that call or compose with it so the original train-time count is active when training resumes.

## Risks / Trade-offs

- Sampling OOM when `--sample_blocks_to_swap 0` is too aggressive -> the restoration path MUST run in `finally`, and docs should recommend using a small positive override if unswapped sampling does not fit.
- Hidden pending transfer state corrupts device layout -> the helper MUST wait for offloader futures before changing counts and before restoring counts.
- Positive sampling override with no train-time offloader is ambiguous -> reject clearly for the first pass instead of silently creating a partial offloader in a surprising state.
- Shared parser flag affects all trainers -> trainer-side use MUST be guarded by model capability checks so unsupported architectures keep existing behavior unless they implement the helper.
