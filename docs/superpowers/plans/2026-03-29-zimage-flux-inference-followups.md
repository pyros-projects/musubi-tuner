# Z-Image And Flux Inference Follow-Ups Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the standalone Z-Image FP8+LoRA inference crash, bring Z-Image prompt-file/API behavior up to Flux parity, and make Flux.2 first-run prompt embedding generation materially faster in prompt-file inference.

**Architecture:** Treat this as three related inference-side follow-ups, not one bundled refactor. First isolate and fix the Z-Image FP8 dtype mismatch at the actual FP8/LoRA boundary. Then bring Z-Image prompt-file and inference CLI behavior into deliberate parity with Flux, including prompt-file LoRA overrides and persistent text embedding caching. Finally, rework Flux prompt embedding precomputation so it uses the same batching mindset as the dedicated cache script, while keeping the current explicit `--save_embeddings` reuse semantics.

**Tech Stack:** Python, PyTorch, safetensors, argparse, unittest, existing Musubi inference helpers

---

## Scope Notes

- This plan intentionally excludes training code.
- This plan intentionally excludes OpenSpec/config/process files.
- This plan covers only standalone inference scripts:
  - `src/musubi_tuner/zimage_generate_image.py`
  - `src/musubi_tuner/flux_2_generate_image.py`
- The dedicated cache script is used as a performance reference, not as a feature target:
  - `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py`

## Known Evidence To Preserve

- Z-Image standalone inference with `--fp8 --fp8_scaled --from_file ... --lora_weight ...` fails at runtime with:
  - `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float`
- The failure occurs in:
  - `src/musubi_tuner/modules/fp8_optimization_utils.py`
  - `fp8_linear_forward_patch(...)`
- The Z-Image model loader already has a suspicious note:
  - `src/musubi_tuner/zimage/zimage_model.py`
  - `# TODO cast weights to mixed precision dtype when fp8_scaled is True, and original weights are in fp32`
- Flux prompt-file inference currently precomputes text prompt embeddings one prompt at a time in:
  - `src/musubi_tuner/flux_2_generate_image.py`
  - `precompute_text_inputs_for_prompts(...)`
- The dedicated Flux cache script already batch-encodes prompts:
  - `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py`
  - `encode_and_save_batch(...)`
- Persistent Flux prompt embedding reuse already works when `--save_embeddings` is provided. Without that flag, only in-process `conds_cache` reuse exists.

## File Map

**Likely Modify**

- `src/musubi_tuner/zimage_generate_image.py`
  - Z-Image CLI parity, prompt-file flow, and any script-level inference wiring needed by the FP8/LoRA fix
- `src/musubi_tuner/zimage/zimage_model.py`
  - Z-Image FP8 load path if the root cause lives in model loading or FP8 monkey-patch setup
- `src/musubi_tuner/modules/fp8_optimization_utils.py`
  - FP8 patched linear forward path if the root cause is dtype promotion during dequantized linear math
- `src/musubi_tuner/flux_2_generate_image.py`
  - Flux prompt embedding batching, cache reuse behavior, prompt-file UX improvements
- `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py`
  - Only if extracting shared batching/cache helpers is cleaner than duplicating logic

**Likely Test**

- `tests/test_zimage_generate_image.py`
  - Extend with FP8+LoRA regression tests and prompt-file CLI parity tests
- `tests/test_flux2_generate_image.py`
  - Extend with batching and persistent cache reuse tests

**Optional New Helper**

- `src/musubi_tuner/utils/prompt_embedding_cache_utils.py`
  - Only create this if Flux script logic becomes duplicated or unreadable. Do not create it by default.

## Chunk 1: Z-Image FP8 + LoRA Correctness

### Task 1: Capture The Z-Image Failure In A Focused Test

**Files:**
- Modify: `tests/test_zimage_generate_image.py`
- Read: `src/musubi_tuner/zimage_generate_image.py`
- Read: `src/musubi_tuner/zimage/zimage_model.py`
- Read: `src/musubi_tuner/modules/fp8_optimization_utils.py`

- [ ] **Step 1: Add a failing regression test for FP8 + LoRA inference dtype handling**

Use a focused test that proves the bad boundary, not a huge end-to-end run. Prefer one of these shapes:

1. A direct `fp8_linear_forward_patch(...)` test with:
   - BF16 input activations
   - FP8-patched module buffers
   - a float32-promoted dequantized weight path
   - expected behavior: no dtype mismatch exception
2. If direct patch testing is too synthetic, a smaller Z-Image loader test that patches enough dependencies to reach the same dtype path without loading the real model.

- [ ] **Step 2: Run the new test alone to verify it fails**

Run:

```bash
./.venv/bin/python -m unittest tests.test_zimage_generate_image -v
```

Expected:
- FAIL on the new FP8+LoRA regression case with a dtype mismatch or equivalent bad-type evidence.

- [ ] **Step 3: Identify the exact root cause before fixing**

Investigate these three hypotheses in order:

1. `self.scale_weight.dtype` or dequantized weight math promotes to `float32`, but `x` remains `bfloat16`
2. LoRA-merged FP8 load path leaves some patched modules in plain `float32` while runtime activations stay `bfloat16`
3. SageAttention is a red herring and the failure happens before attention backend code runs

Record the confirmed root cause directly in the code comment or commit message, not just in chat.

- [ ] **Step 4: Implement the smallest root-cause fix**

Allowed fix locations:
- `src/musubi_tuner/modules/fp8_optimization_utils.py`
- `src/musubi_tuner/zimage/zimage_model.py`

Do not:
- add a broad “cast everything to float32” workaround
- special-case `sageattn` unless evidence proves the backend changes dtype semantics
- bundle unrelated FP8 cleanup

- [ ] **Step 5: Re-run the focused test**

Run:

```bash
./.venv/bin/python -m unittest tests.test_zimage_generate_image -v
```

Expected:
- PASS for the new regression case

- [ ] **Step 6: Run a quick script-level smoke test with the user’s reproduction command**

Run a shortened local command if possible, or the exact command if runtime cost is acceptable. Capture whether:
- prompt-file mode still works
- LoRA still loads
- generated latents/images no longer fail before decode

- [ ] **Step 7: Commit**

```bash
git add tests/test_zimage_generate_image.py src/musubi_tuner/modules/fp8_optimization_utils.py src/musubi_tuner/zimage/zimage_model.py src/musubi_tuner/zimage_generate_image.py
git commit -m "fix: preserve dtype in zimage fp8 lora inference"
```

## Chunk 2: Z-Image Full Prompt-File/API Parity With Flux

### Task 2: Add The Full Z-Image Inference Conveniences We Want

**Files:**
- Modify: `src/musubi_tuner/zimage_generate_image.py`
- Test: `tests/test_zimage_generate_image.py`
- Reference: `src/musubi_tuner/flux_2_generate_image.py`

- [ ] **Step 1: Lock the parity surface**

Required parity items:
- `--sample_prompts` as an alias for `--from_file`
- `--save_strategy` with the same `immediate` vs `deferred` meaning Flux has
- `--save_embeddings` directory support for persistent text embedding reuse across runs
- prompt-file per-line `--lora_weight` and `--lora_multiplier` overrides
- prompt-file parsing for lines that begin with `--` and carry only overrides
- `apply_overrides(...)` normalization of prompt-level LoRA args
- immediate-save prompt-file processing path so long prompt files do not require holding every latent until the end
- the same batch consistency rule Flux currently enforces when mixed prompt-level LoRA specs would force a DiT reload

- [ ] **Step 2: Add failing parser, cache, and control-flow tests**

Add tests for:
- `--sample_prompts` alias parsing
- `--save_strategy immediate` dispatching to an immediate-save path
- `--save_strategy deferred` keeping the current batched save behavior
- prompt-file line parsing for `--lora_weight` and `--lora_multiplier`
- prompt-file line parsing when the line starts with `--` and has no prompt text
- `apply_overrides(...)` normalizing prompt-level LoRA args
- persistent embedding cache round-trip for Z-Image text embeddings
- second-run cache hit behavior when `--save_embeddings` is used

- [ ] **Step 3: Run the targeted Z-Image tests to confirm they fail**

Run:

```bash
./.venv/bin/python -m unittest tests.test_zimage_generate_image -v
```

Expected:
- FAIL on new parser/dispatch assertions

- [ ] **Step 4: Implement the full approved parity set**

Likely additions in `src/musubi_tuner/zimage_generate_image.py`:
- parser alias for `--sample_prompts`
- `--save_strategy` argument
- `--save_embeddings` argument
- `process_prompts_with_immediate_save(...)`
- prompt embedding cache key/load/save helpers
- prompt-file `lora_weight` / `lora_multiplier` parsing
- LoRA normalization in `apply_overrides(...)`
- any small shared batching/cache helper needed to keep code readable

Keep behavior aligned with Flux where it helps, but do not force the two scripts into a premature shared abstraction. Matching API and prompt-file semantics matters more than deduplicating code right now.

- [ ] **Step 5: Re-run the Z-Image test file**

Run:

```bash
./.venv/bin/python -m unittest tests.test_zimage_generate_image -v
```

Expected:
- PASS

- [ ] **Step 6: Commit**

```bash
git add src/musubi_tuner/zimage_generate_image.py tests/test_zimage_generate_image.py
git commit -m "feat: add zimage prompt-file inference parity"
```

## Chunk 3: Flux Prompt Embedding Throughput

### Task 3: Replace Per-Prompt Text Encoding With Batched Precompute

**Files:**
- Modify: `src/musubi_tuner/flux_2_generate_image.py`
- Read: `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py`
- Test: `tests/test_flux2_generate_image.py`

- [ ] **Step 1: Add a failing batching test**

Test shape:
- prompt-file inference with multiple prompts
- mocked text embedder
- expected behavior after fix: batched precompute should call the embedder on grouped prompt lists rather than one prompt at a time

This does not need a wall-clock benchmark. Count calls and inspect argument shapes instead.

- [ ] **Step 2: Run the Flux test file and confirm the new batching test fails**

Run:

```bash
./.venv/bin/python -m unittest tests.test_flux2_generate_image -v
```

Expected:
- FAIL because the current code loops through `prepare_text_inputs(...)` per prompt

- [ ] **Step 3: Implement batched text precompute**

Recommended design:
- add a new helper in `src/musubi_tuner/flux_2_generate_image.py` that:
  - collects prompt strings needed for the batch
  - deduplicates repeated prompt text and repeated negative prompts
  - checks persistent cache first when `--save_embeddings` is enabled
  - batch-encodes only the cache misses in one or a small number of embedder calls
  - maps results back to per-prompt `ctx_vec` / `negative_ctx_vec`

Avoid:
- rewriting the cache script
- introducing a new shared helper file unless the local function becomes genuinely unwieldy

- [ ] **Step 4: Re-run the Flux test file**

Run:

```bash
./.venv/bin/python -m unittest tests.test_flux2_generate_image -v
```

Expected:
- PASS for the new batching behavior

- [ ] **Step 5: Do a manual performance sanity check**

Use a small prompt file and compare:
- before/after text precompute logs
- number of embedder calls
- rough elapsed time for the precompute phase

Do not promise “100s/minute” unless the measurement supports it.

- [ ] **Step 6: Commit**

```bash
git add src/musubi_tuner/flux_2_generate_image.py tests/test_flux2_generate_image.py
git commit -m "perf: batch flux prompt embedding precompute"
```

## Chunk 4: Final Verification And User-Facing Notes

### Task 4: Verify The Combined Inference Surface

**Files:**
- Modify if needed: `src/musubi_tuner/zimage_generate_image.py`
- Modify if needed: `src/musubi_tuner/flux_2_generate_image.py`
- Test: `tests/test_zimage_generate_image.py`
- Test: `tests/test_flux2_generate_image.py`

- [ ] **Step 1: Run the focused inference regression suite**

Run:

```bash
./.venv/bin/python -m unittest tests.test_zimage_generate_image tests.test_flux2_generate_image -v
```

Expected:
- PASS

- [ ] **Step 2: Run the broader migration safety net**

Run:

```bash
./.venv/bin/python -m unittest \
  tests.test_remaining_migration_ports \
  tests.test_compile_prewarm \
  tests.test_sampling_adapter_compile \
  tests.test_ltx2_sampling_lora \
  tests.test_ltx2_lora_presets \
  tests.test_zimage_generate_image \
  tests.test_flux2_generate_image
```

Expected:
- PASS

- [ ] **Step 3: Manually smoke test the real scripts**

Minimum manual checks:
- Z-Image prompt-file inference with `--fp8 --fp8_scaled --lora_weight ...`
- Flux prompt-file inference with and without `--save_embeddings`
- Flux second-run reuse behavior confirmed via logs

- [ ] **Step 4: Update help text or docs only if behavior changed**

Only document:
- new Z-Image parser options
- final Flux prompt cache semantics

- [ ] **Step 5: Final commit**

```bash
git add src/musubi_tuner/zimage_generate_image.py src/musubi_tuner/zimage/zimage_model.py src/musubi_tuner/modules/fp8_optimization_utils.py src/musubi_tuner/flux_2_generate_image.py tests/test_zimage_generate_image.py tests/test_flux2_generate_image.py
git commit -m "fix: improve zimage and flux inference followups"
```

## Review Points For Human Approval

Before execution, review these decisions:

1. Z-Image prompt-file LoRA behavior:
   - match Flux exactly and reject mixed prompt-level LoRA specs inside one batch, or
   - go beyond Flux and group prompts by LoRA spec so a single prompt file can drive multiple DiT loads
2. Z-Image embedding cache scope:
   - strict Flux-style prompt-only cache, or
   - include any additional Z-Image text-encoder settings in the hash if needed by implementation
3. Performance target:
   - “materially faster than current prompt-at-a-time path” is realistic
   - “match dedicated cache script throughput exactly” may not be realistic because inference still does more than pure text encoding

## Expected Deliverables

- Z-Image standalone inference no longer crashes on FP8+LoRA prompt-file runs
- Z-Image prompt-file inference matches Flux’s API surface, including prompt-file LoRA overrides and `--save_embeddings`
- Flux prompt-file inference batches text embedding work instead of encoding prompts one by one
- Regression tests cover all of the above

Plan complete and saved to `docs/superpowers/plans/2026-03-29-zimage-flux-inference-followups.md`. Ready to execute?
