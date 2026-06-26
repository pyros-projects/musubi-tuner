## 1. TDD Coverage

- [ ] 1.1 Add CPU tests for Boogu `enable_block_swap` count validation and offloader construction.
- [ ] 1.2 Add CPU tests that positive Boogu `--blocks_to_swap` is accepted by trainer argument handling.
- [ ] 1.3 Add CPU tests for `move_to_device_except_swap_blocks` excluding all five Boogu block containers from the bulk move.
- [ ] 1.4 Add CPU tests for wait/submit ordering across context, noise, reference-image, double-stream, and single-stream stages.
- [ ] 1.5 Add CPU tests for `switch_block_swap_for_inference`, `switch_block_swap_for_training`, and `_wait_for_pending_block_swaps`.
- [ ] 1.6 Add CPU tests for `override_block_swap_for_sampling` positive, zero, invalid, and failure-restoration cases.

## 2. Transformer Block-Swap Surface

- [ ] 2.1 Add an ordered block-list helper that returns Boogu blocks in actual forward order without registering duplicate modules.
- [ ] 2.2 Implement `enable_block_swap` with `ModelOffloader`, computed block-count validation, backward-support wiring, and pinned-memory support.
- [ ] 2.3 Implement `move_to_device_except_swap_blocks` by temporarily detaching `context_refiner`, `noise_refiner`, `ref_image_refiner`, `double_stream_layers`, and `single_stream_layers`.
- [ ] 2.4 Implement `prepare_block_swap_before_forward`, `_wait_for_pending_block_swaps`, `switch_block_swap_for_inference`, and `switch_block_swap_for_training`.
- [ ] 2.5 Implement `override_block_swap_for_sampling` with Krea2-equivalent validation, prepare, wait, and restore behavior.

## 3. Forward Integration

- [ ] 3.1 Add a helper for running a block with optional wait, checkpointed execution, and submit using a global block index.
- [ ] 3.2 Thread global block indices through the context-refiner loop.
- [ ] 3.3 Thread global block indices through `img_patch_embed_and_refine` for noise and reference-image refiners.
- [ ] 3.4 Thread global block indices through double-stream and single-stream loops, preserving tuple outputs for double-stream blocks.
- [ ] 3.5 Keep the disabled-block-swap path equivalent to the current no-op behavior.

## 4. Trainer Integration And Verification

- [ ] 4.1 Remove the Boogu trainer guard that rejects positive `--blocks_to_swap`.
- [ ] 4.2 Ensure shared `--sample_blocks_to_swap` handling reaches Boogu through `override_block_swap_for_sampling`.
- [ ] 4.3 Run `uv run pytest tests/test_boogu_image_support.py -q` or the dedicated Boogu block-swap test file if split out.
- [ ] 4.4 Run `openspec validate add-boogu-block-swap --strict`.
- [ ] 4.5 Record that real GPU block-swap smoke is deferred until GPU capacity is available.
