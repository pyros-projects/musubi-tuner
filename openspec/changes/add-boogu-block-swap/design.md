## Context

Boogu's transformer currently sets `blocks_to_swap = 0` and exposes the shared methods `switch_block_swap_for_inference`, `switch_block_swap_for_training`, `move_to_device_except_swap_blocks`, and `prepare_block_swap_before_forward` as no-ops. This was enough to keep shared sampling from crashing, but the trainer still rejects positive `--blocks_to_swap`.

Other Musubi model families use `musubi_tuner.modules.custom_offloading_utils.ModelOffloader`: the trainer calls `enable_block_swap(...)`, then `move_to_device_except_swap_blocks(...)`, and the transformer forward path waits for each block before execution and submits post-execution movement for streaming CPU/GPU offload. Krea2 also exposes `override_block_swap_for_sampling(...)` so `--sample_blocks_to_swap` can temporarily alter block-swap behavior during snapshot sampling.

Boogu differs from Krea2 because its forward pass does not have one simple `self.blocks` list. The execution order is:

1. `context_refiner`
2. `noise_refiner`
3. `ref_image_refiner`
4. `double_stream_layers`
5. `single_stream_layers`

For the default Boogu Base config, this is 2 context + 2 noise + 2 reference-image + 8 double-stream + 32 single-stream blocks, for 46 offloadable blocks. The implementation must keep that order without registering duplicate module aliases that would change the state dict.

## Goals / Non-Goals

**Goals:**

- Allow Boogu training to use positive `--blocks_to_swap`.
- Stream Boogu block movement around all transformer block groups in actual forward order.
- Keep swapped block weights CPU-resident during `move_to_device_except_swap_blocks`.
- Support `switch_block_swap_for_inference` and `switch_block_swap_for_training` for shared training/sampling transitions.
- Support `override_block_swap_for_sampling` so `--sample_blocks_to_swap` works for Boogu.
- Test the behavior on CPU with fake offloaders/modules where possible.

**Non-Goals:**

- Do not add a new offloader implementation.
- Do not implement Boogu Edit/TI2I block swap.
- Do not change Boogu attention math, LoRA targets, fp8 quantization, or snapshot sampling quality.
- Do not require a real GPU smoke test in the automated verification pass.

## Decisions

### Decision: Use a plain ordered block list, not a registered ModuleList alias

Boogu should expose a helper such as `_block_swap_blocks()` that returns:

`list(context_refiner) + list(noise_refiner) + list(ref_image_refiner) + list(double_stream_layers) + list(single_stream_layers)`

This list is passed to `ModelOffloader` and to submit calls. It must remain a plain Python list so the same modules are not registered twice in the transformer.

Alternatives considered:

- Add `self.layers = nn.ModuleList(...)`: rejected because duplicate module registration can alter `state_dict()` and loading behavior.
- Swap only double/single blocks: rejected because the current Boogu forward path also spends memory in refiner blocks and a partial list would make `blocks_to_swap` counts misleading.

### Decision: Validate counts from the computed ordered list

`enable_block_swap` should compute `num_blocks = len(_block_swap_blocks())` and reject requests above a conservative maximum, following the existing Musubi pattern of leaving at least two blocks resident. For default Boogu Base, that means at most 44 swapped blocks out of 46.

Alternatives considered:

- Reuse only `num_layers - 2`: rejected because Boogu has extra refiner stages outside `num_layers`.
- Allow all blocks to swap: rejected because existing model-family implementations keep some blocks resident and this is the safer parity path.

### Decision: Thread stable global block indices through each loop

Each Boogu loop should call the same wait/run/submit wrapper with a stable global block index. The indices should advance in forward order across helper boundaries, including `img_patch_embed_and_refine(...)`. The double-stream wrapper must support tuple outputs, while the other stages return a single tensor.

Alternatives considered:

- Create separate offloaders per stage: rejected for the first pass because shared `--blocks_to_swap` semantics are easiest to reason about as one ordered block budget.

### Decision: Temporarily detach all block ModuleLists during device moves

`move_to_device_except_swap_blocks(device)` should temporarily replace the five block `ModuleList` attributes with empty `ModuleList`s, call `self.to(device)` for the rest of the model, then restore the original lists. This mirrors other model-family patterns while accounting for Boogu's multiple block containers.

Alternatives considered:

- Move every module and then move swapped blocks back to CPU: rejected because it increases peak VRAM during model load.

### Decision: Match Krea2's sampling override lifecycle

`override_block_swap_for_sampling(sample_blocks_to_swap, device)` should validate the requested count, wait for pending transfers, update `self.blocks_to_swap` and the offloader count for the sampling context, prepare block devices when the target is positive, and restore the original state in `finally`.

Alternatives considered:

- Ignore `--sample_blocks_to_swap` for Boogu: rejected because the shared sampler already exposes this control and users expect parity with other large models.

## Risks / Trade-offs

- Block indices can drift if helper loops are refactored later -> centralize offset calculation and add CPU tests for the observed wait/submit order.
- Reference-image refiners may run even for Base T2I with empty reference images -> include them in the ordered list anyway so counts and forward order stay stable.
- Offloader futures can remain pending when sampling or decode offload begins -> expose `_wait_for_pending_block_swaps()` and use the existing helper contract from other models.
- FP8 monkey-patched linear layers plus block swap need real GPU validation later -> keep automated tests CPU-focused and mark GPU smoke deferred until available.
- `torch.compile` and prewarm behavior may interact with module movement -> do not change compile behavior in this pass; test import/setup and leave heavy smoke for GPU availability.

## Migration Plan

1. Add failing CPU tests for Boogu block-swap setup, validation, and trainer acceptance.
2. Add failing CPU tests for `move_to_device_except_swap_blocks` preserving the block ModuleLists while moving non-block modules.
3. Add failing CPU tests for forward wait/submit ordering across the ordered block groups.
4. Add failing CPU tests for `switch_block_swap_for_inference`, `switch_block_swap_for_training`, `_wait_for_pending_block_swaps`, and `override_block_swap_for_sampling`.
5. Implement the ordered block-list helper and real `ModelOffloader` methods.
6. Thread global block indices through Boogu forward and refiner helper loops.
7. Remove the trainer rejection for positive `--blocks_to_swap`.
8. Run CPU tests and defer real GPU smoke until the GPU is free.

Rollback is file-level: restore the Boogu no-op block-swap methods and trainer guard. No checkpoint or data migration is required.

## Open Questions

- None for the initial implementation. Real training smoke should verify memory and speed once the GPU is available.
