# LTX Prompt-File Per-Prompt LoRA Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add prompt-file-native per-prompt LoRA stacks for LTX so TOML `--sample_prompts` files can define root baseline LoRAs plus per-subset `loras` with explicit `lora_mode = "extend" | "replace"` semantics.

**Architecture:** Preserve the current LTX prompt-file flow, but add an LTX-specific prompt normalization layer that resolves effective per-sample LoRA stacks into `resolved_loras`. Reuse the existing reversible sampling-LoRA runtime path by moving prompt-file LoRA application from global pre-loop setup to per-sample apply/restore, while keeping distilled two-stage LoRA handling separate.

**Tech Stack:** Python 3, argparse, TOML prompt files, PyTorch, existing Musubi sampling-LoRA runtime in `hv_train_network.py`, LTX trainer/inference code, `unittest`

---

## File Map

**Create**
- `src/musubi_tuner/ltx2_prompt_lora_utils.py`
  - LTX-specific prompt-file parsing and validation for root `loras`, subset `loras`, and `lora_mode`
  - helpers to resolve effective per-sample LoRA stacks into `resolved_loras`
- `tests/test_ltx2_prompt_lora_utils.py`
  - focused unit tests for parsing, validation, and `extend` / `replace` resolution

**Modify**
- `src/musubi_tuner/ltx2_train_network.py`
  - route LTX TOML prompt loading through the new helper
  - carry `resolved_loras` into `sample_parameters`
  - apply and restore prompt-local LoRA stacks per sample in both standalone preview and training preview paths
- `src/musubi_tuner/hv_train_network.py`
  - extract reusable sampling-LoRA application helpers that accept explicit specs instead of only CLI-global args
  - preserve current global sampling-LoRA behavior on top of the new lower-level helper
- `src/musubi_tuner/ltx2_generate_video.py`
  - stop doing one-time global pre-merge when prompt-file-driven per-sample LoRAs are active
  - fold CLI `--lora_weight/--lora_multiplier` into the prompt baseline semantics for prompt-file mode
- `docs/config_examples/ltx2_sample_prompts_two_stage.toml`
  - add the new prompt-file shape with root `loras`, subset `loras`, and `lora_mode`
- `docs/ltx_2.md`
  - document the feature, prompt-file schema, precedence, and warnings
- `tests/test_ltx2_sampling_lora.py`
  - add integration-style tests proving prompt A and prompt B can use different LoRA stacks without leakage

**Reference During Implementation**
- `src/musubi_tuner/hv_train.py`
  - existing generic `load_prompts()` behavior and TOML flattening limits
- `src/musubi_tuner/ltx2_inference.py`
  - current distilled LoRA runtime switching and restore logic
- `docs/superpowers/specs/2026-03-30-ltx-prompt-file-lora-design.md`
  - approved design source of truth

---

## Chunk 1: Prompt Parsing And Resolution

### Task 1: Add LTX-specific prompt-file LoRA utilities

**Files:**
- Create: `src/musubi_tuner/ltx2_prompt_lora_utils.py`
- Test: `tests/test_ltx2_prompt_lora_utils.py`

- [ ] **Step 1: Write the failing parsing and resolution tests**

Add tests covering:
- root-only baseline `loras`
- subset `loras` with omitted `lora_mode` defaulting to `extend`
- subset `lora_mode = "replace"`
- subset without `loras` inheriting baseline only
- validation failures for bad `lora_mode`, bad `loras`, missing `path`, and invalid `weight`

Example test skeleton:

```python
def test_resolve_subset_loras_extend_merges_root_and_subset():
    data = {
        "prompt": {
            "width": 1280,
            "loras": [{"path": "/a.safetensors", "weight": 0.3, "merge": False}],
            "subset": [
                {
                    "prompt": "hello",
                    "lora_mode": "extend",
                    "loras": [{"path": "/b.safetensors", "weight": 0.8, "merge": False}],
                }
            ],
        }
    }

    prompts = resolve_ltx_prompt_file_data(data)
    assert prompts[0]["resolved_loras"] == [
        {"path": "/a.safetensors", "weight": 0.3, "merge": False},
        {"path": "/b.safetensors", "weight": 0.8, "merge": False},
    ]
```

- [ ] **Step 2: Run the new test module to verify it fails**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest tests.test_ltx2_prompt_lora_utils -v
```

Expected:
- failure because `tests/test_ltx2_prompt_lora_utils.py` or `resolve_ltx_prompt_file_data` does not exist yet

- [ ] **Step 3: Implement the minimal utility module**

Add a focused helper module with:
- `validate_ltx_prompt_lora_entry(entry)`
- `normalize_ltx_prompt_lora_entries(entries)`
- `resolve_ltx_prompt_file_data(data, *, baseline_loras=None)`
- optional file wrapper like `load_ltx_prompt_file_with_resolved_loras(prompt_file, *, baseline_loras=None)`

Implementation requirements:
- preserve existing prompt scalar fields
- attach `resolved_loras` to each resolved subset prompt dict
- default subset `lora_mode` to `extend` if subset `loras` exist and mode is omitted
- allow no `loras` at all
- keep output compatible with existing `sample_parameters` dictionaries

- [ ] **Step 4: Re-run the new parsing test module**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest tests.test_ltx2_prompt_lora_utils -v
```

Expected:
- PASS for all new parsing and validation tests

- [ ] **Step 5: Commit the parsing utility slice**

```bash
git add src/musubi_tuner/ltx2_prompt_lora_utils.py tests/test_ltx2_prompt_lora_utils.py
git commit -m "feat: add LTX prompt-file LoRA resolution helpers"
```

### Task 2: Route LTX prompt normalization through the new resolver

**Files:**
- Modify: `src/musubi_tuner/ltx2_train_network.py`
- Test: `tests/test_ltx2_prompt_lora_utils.py`

- [ ] **Step 1: Add a failing trainer-facing test or expand utility tests to cover trainer-compatible output**

Add a test that verifies the resolved prompt dict includes:
- existing scalar fields like `width`, `height`, `frame_count`
- `enum`
- `resolved_loras`

- [ ] **Step 2: Run that test to verify current trainer loading is insufficient**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest tests.test_ltx2_prompt_lora_utils -v
```

Expected:
- failure until `ltx2_train_network.py` uses the new LTX-specific resolution path

- [ ] **Step 3: Update `ltx2_train_network.py` prompt processing**

Implementation shape:
- when `sample_prompts` ends with `.toml`, use the new LTX-specific resolver instead of raw generic `load_prompts()`
- when `sample_prompts` is `.txt` or `.json`, preserve current behavior
- fold standalone CLI `args.lora_weight` / `args.lora_multiplier` into prompt baseline resolution only for prompt-file mode
- keep prompt dicts compatible with `_apply_sample_defaults()`

- [ ] **Step 4: Run the parsing and trainer-normalization tests again**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest tests.test_ltx2_prompt_lora_utils -v
```

Expected:
- PASS

- [ ] **Step 5: Commit the trainer normalization slice**

```bash
git add src/musubi_tuner/ltx2_train_network.py tests/test_ltx2_prompt_lora_utils.py
git commit -m "feat: resolve per-prompt LTX LoRA stacks from TOML"
```

---

## Chunk 2: Per-Sample Runtime Application

### Task 3: Extract explicit sampling-LoRA spec helpers from the shared runtime

**Files:**
- Modify: `src/musubi_tuner/hv_train_network.py`
- Test: `tests/test_ltx2_sampling_lora.py`

- [ ] **Step 1: Add a failing test for applying explicit LoRA specs**

Add a test that exercises a new helper contract like:

```python
applied = trainer._apply_sampling_lora_specs(
    transformer,
    device,
    specs=[("/tmp/a.safetensors", 0.5), ("/tmp/b.safetensors", 1.0)],
)
```

and verifies:
- the requested weights are loaded via cache-aware helpers
- overlays or backups are returned
- restoration can cleanly undo them

- [ ] **Step 2: Run the targeted sampling test to verify failure**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest tests.test_ltx2_sampling_lora -v
```

Expected:
- failure because `_apply_sampling_lora_specs` does not exist yet

- [ ] **Step 3: Implement explicit-spec apply/restore helpers in `hv_train_network.py`**

Refactor:
- keep `_get_sampling_lora_specs(args)` for old CLI-global sampling LoRA behavior
- add lower-level helpers that accept explicit specs directly
- implement them in terms of existing:
  - `_load_sampling_lora_weights`
  - `_apply_sampling_lora_network`
  - `_restore_sampling_lora_network`
  - `_clear_sampling_lora_runtime_overlays`

Target API:
- `_apply_sampling_lora_specs(args, transformer, device, specs)`
- `_restore_sampling_lora_specs(args, transformer, device, applied)`

- [ ] **Step 4: Make old global sampling-LoRA behavior call the new helper**

Keep current behavior intact by having `_apply_sampling_lora()` delegate to `_apply_sampling_lora_specs()` with `_get_sampling_lora_specs(args)`.

- [ ] **Step 5: Re-run the targeted sampling test**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest tests.test_ltx2_sampling_lora -v
```

Expected:
- PASS for the new helper tests

- [ ] **Step 6: Commit the shared runtime refactor**

```bash
git add src/musubi_tuner/hv_train_network.py tests/test_ltx2_sampling_lora.py
git commit -m "refactor: add explicit sampling LoRA spec helpers"
```

### Task 4: Apply prompt-local LoRA stacks per sample in LTX preview sampling

**Files:**
- Modify: `src/musubi_tuner/ltx2_train_network.py`
- Test: `tests/test_ltx2_sampling_lora.py`

- [ ] **Step 1: Add failing tests for prompt-local LoRA application in the sample loop**

Add tests that mock the apply/restore helper and verify:
- prompt A gets stack A
- prompt B gets stack B
- restoration happens between prompts
- no stack is applied when `resolved_loras` is empty

- [ ] **Step 2: Run the targeted sampling test module**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest tests.test_ltx2_sampling_lora -v
```

Expected:
- failure until per-sample prompt-local application is wired in

- [ ] **Step 3: Implement prompt-local apply/restore inside `ltx2_train_network.sample_images()`**

Implementation requirements:
- compute prompt-local specs from `sample_parameter["resolved_loras"]`
- apply them immediately before `sample_image_inference()`
- restore them in a `finally` block after each prompt
- preserve current offload and cleanup behavior
- do not interfere with distilled LoRA handling inside `LTX2Inferencer`

- [ ] **Step 4: Re-run the targeted sampling test module**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest tests.test_ltx2_sampling_lora -v
```

Expected:
- PASS for prompt-local apply/restore tests

- [ ] **Step 5: Commit the LTX preview sampling slice**

```bash
git add src/musubi_tuner/ltx2_train_network.py tests/test_ltx2_sampling_lora.py
git commit -m "feat: apply prompt-local LoRA stacks during LTX sampling"
```

### Task 5: Remove one-time standalone global pre-merge for prompt-file LoRA runs

**Files:**
- Modify: `src/musubi_tuner/ltx2_generate_video.py`
- Test: `tests/test_ltx2_sampling_lora.py`

- [ ] **Step 1: Add a failing standalone-path test**

Add a test proving:
- single prompt CLI mode can still use global `--lora_weight`
- prompt-file mode does not do one-time pre-merge when `resolved_loras` will be applied per sample

- [ ] **Step 2: Run the targeted sampling test module**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest tests.test_ltx2_sampling_lora -v
```

Expected:
- failure until the standalone flow distinguishes single-prompt global merge from prompt-file per-sample mode

- [ ] **Step 3: Update standalone generation control flow**

Implementation requirements:
- keep current one-time `_merge_lora_weights()` for direct single-prompt CLI inference
- skip one-time pre-merge for prompt-file TOML mode when `resolved_loras` are present
- ensure CLI `--lora_weight/--lora_multiplier` feed baseline resolution for prompt-file mode

- [ ] **Step 4: Re-run the targeted sampling test module**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest tests.test_ltx2_sampling_lora -v
```

Expected:
- PASS

- [ ] **Step 5: Commit the standalone integration slice**

```bash
git add src/musubi_tuner/ltx2_generate_video.py tests/test_ltx2_sampling_lora.py
git commit -m "feat: use prompt-local LoRA stacks in standalone LTX prompt files"
```

---

## Chunk 3: Docs, Examples, And Verification

### Task 6: Update docs and example prompt files

**Files:**
- Modify: `docs/config_examples/ltx2_sample_prompts_two_stage.toml`
- Modify: `docs/ltx_2.md`

- [ ] **Step 1: Update the example TOML to demonstrate baseline and per-subset LoRAs**

Document:
- root `[prompt].loras`
- subset `lora_mode = "extend"`
- subset `lora_mode = "replace"`
- a subset with no prompt-local `loras`

- [ ] **Step 2: Update the LTX docs**

Document:
- schema for `loras` entries
- `lora_mode` semantics
- precedence with CLI `--lora_weight/--lora_multiplier`
- recommendation to prefer `merge = false`
- note that distilled two-stage LoRA is separate from prompt-file LoRA stacks

- [ ] **Step 3: Commit the docs slice**

```bash
git add docs/config_examples/ltx2_sample_prompts_two_stage.toml docs/ltx_2.md
git commit -m "docs: add prompt-file per-prompt LoRA examples for LTX"
```

### Task 7: Run focused verification

**Files:**
- Test: `tests/test_ltx2_prompt_lora_utils.py`
- Test: `tests/test_ltx2_sampling_lora.py`

- [ ] **Step 1: Run the new prompt-file utility tests**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest tests.test_ltx2_prompt_lora_utils -v
```

Expected:
- PASS

- [ ] **Step 2: Run the focused LTX sampling LoRA tests**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest tests.test_ltx2_sampling_lora -v
```

Expected:
- PASS

- [ ] **Step 3: Run the repo’s recommended focused Northstar/LTX regression slice if any touched areas overlap**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest discover -s tests -p 'test_ltx2*.py' -v
```

Expected:
- PASS, or explicit list of unrelated pre-existing failures if the suite is not currently clean

- [ ] **Step 4: Inspect the sample example file visually for correctness**

Check:
- `docs/config_examples/ltx2_sample_prompts_two_stage.toml`
- no stale comments describing the old capabilities only
- examples align with the documented semantics

- [ ] **Step 5: Commit final verification or follow-up fixes**

```bash
git add tests/test_ltx2_prompt_lora_utils.py tests/test_ltx2_sampling_lora.py docs/config_examples/ltx2_sample_prompts_two_stage.toml docs/ltx_2.md src/musubi_tuner/ltx2_prompt_lora_utils.py src/musubi_tuner/ltx2_train_network.py src/musubi_tuner/hv_train_network.py src/musubi_tuner/ltx2_generate_video.py
git commit -m "test: verify LTX prompt-file per-prompt LoRA support"
```

---

## Notes For The Implementer

- Do not mix prompt-file concept/style LoRAs into the existing distilled two-stage LoRA control flow.
- Keep prompt-file LoRA resolution LTX-specific in v1; avoid “improving” generic prompt loading for other families in the same change.
- Prefer adding a small utility module over bloating `ltx2_train_network.py` further.
- Preserve current behavior for users who do not use prompt-file `loras`.
- Treat `merge = false` as the primary path to optimize and validate.

Plan complete and saved to `docs/superpowers/plans/2026-03-30-ltx-prompt-file-lora.md`. Ready to execute?
