# LTX Prompt-File Image Pool Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add prompt-file-native image-pool assignment for LTX TOML sample prompts so root image sources can resolve to per-subset `image_path` values while still allowing plain T2V subsets.

**Architecture:** Keep all new semantics in the LTX TOML resolver. The resolver should translate root-level image pool config plus per-subset overrides into ordinary resolved prompt dicts that either contain `image_path` or omit it entirely. Sampling, preview generation, and sample-latent precache should continue consuming the existing `image_path` contract without learning any new pool-specific behavior.

**Tech Stack:** Python 3, `toml`, `pathlib`, `unittest`, existing LTX prompt-file resolver and sampling pipeline.

---

## File Structure

### Existing files to modify

- `src/musubi_tuner/ltx2_prompt_lora_utils.py`
  - Expand from LoRA-only normalization into full LTX prompt-file resolution for image pools, subset opt-out, and filesystem validation.
- `tests/test_ltx2_prompt_lora_utils.py`
  - Add resolver-focused unit tests for pool construction, assignment rules, and validation errors.
- `tests/test_ltx2_cache_text_encoder_outputs.py`
  - Add one integration-oriented check that TOML sample-prompt precache sees resolved `image_path` values from the new resolver behavior.
- `docs/ltx_2.md`
  - Document the new root-level image pool fields, subset opt-out behavior, and failure semantics.
- `docs/config_examples/ltx2_sample_prompts_two_stage.toml`
  - Add a concrete mixed T2V/I2V example showing root pool usage, explicit `image_path`, and `use_image_pool = false`.

### No runtime sampler changes expected

- `src/musubi_tuner/ltx2_train_network.py`
- `src/musubi_tuner/ltx2_cache_text_encoder_outputs.py`

These should keep working through existing `load_ltx_prompt_file_with_resolved_loras(...)` usage once the resolver emits final `image_path` values.

---

## Chunk 1: Resolver Tests First

### Task 1: Add failing resolver tests for image-pool semantics

**Files:**
- Modify: `tests/test_ltx2_prompt_lora_utils.py`

- [ ] **Step 1: Add a unique-assignment happy-path test**

Add a test that:

- creates a temporary directory with at least two image files
- defines a root `[prompt]` with `input_images_dir`
- defines multiple subsets with no explicit `image_path`
- asserts each eligible subset gets a distinct resolved `image_path`

Use a structure like:

```python
def test_root_input_images_dir_assigns_unique_images_to_subsets(self):
    data = {
        "prompt": {
            "input_images_dir": str(images_dir),
            "subset": [
                {"prompt": "Prompt A"},
                {"prompt": "Prompt B"},
            ],
        }
    }
    prompts = resolve_ltx_prompt_file_data(data)
    self.assertEqual(prompts[0]["image_path"], str(img_a))
    self.assertEqual(prompts[1]["image_path"], str(img_b))
```

- [ ] **Step 2: Add a random-order assignment test**

Use `unittest.mock.patch` on `random.shuffle` so the test is deterministic. Verify that `image_input_order = "random"` causes the shuffled order to be used for assignment.

- [ ] **Step 3: Add explicit override and T2V opt-out tests**

Add separate tests for:

- subset `image_path` taking precedence over pool assignment
- subset `use_image_pool = false` producing no `image_path`
- mixed prompt file with one pool-assigned subset, one explicit-image subset, and one T2V subset

- [ ] **Step 4: Add validation error tests**

Add failing tests for:

- invalid `image_input_order`
- invalid `image_assignment`
- nonexistent `input_images_dir`
- nonexistent explicit `image_path`
- too few images for `image_assignment = "unique"`
- root pool resolving to zero supported image files when a subset requires pool assignment

- [ ] **Step 5: Run just the resolver tests to verify they fail**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest -v tests.test_ltx2_prompt_lora_utils
```

Expected:

- new tests fail because image-pool fields are not implemented yet
- existing prompt-file LoRA tests may still pass

- [ ] **Step 6: Commit the red tests**

```bash
git add tests/test_ltx2_prompt_lora_utils.py
git commit -m "test: add ltx prompt image pool resolver coverage"
```

---

## Chunk 2: Implement Resolver Semantics

### Task 2: Extend the LTX TOML resolver to resolve image pools into plain `image_path`

**Files:**
- Modify: `src/musubi_tuner/ltx2_prompt_lora_utils.py`
- Test: `tests/test_ltx2_prompt_lora_utils.py`

- [ ] **Step 1: Add image-config validation helpers**

Implement focused helpers in `src/musubi_tuner/ltx2_prompt_lora_utils.py` for:

- `input_images`
- `input_images_dir`
- `image_input_order`
- `image_assignment`
- subset `use_image_pool`
- subset `image_path`

Keep them small and return normalized values rather than mutating globals.

Suggested helpers:

```python
def normalize_ltx_prompt_image_inputs(...)
def collect_ltx_prompt_pool_images(...)
def resolve_subset_image_mode(...)
```

- [ ] **Step 2: Add supported-image file collection**

Collect candidate images from:

- root `input_images`
- recursively from `input_images_dir`

Use `pathlib.Path` and a fixed supported-extension set such as:

```python
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
```

Non-image files in the directory should be ignored, not treated as hard errors.

- [ ] **Step 3: Resolve per-subset assignment rules**

Update `resolve_ltx_prompt_file_data(...)` so it:

- resolves combined baseline LoRAs as it does today
- resolves root image pool config once
- determines which subsets are eligible for pool assignment
- assigns one image per eligible subset for `image_assignment = "unique"`
- writes final `image_path` into each resolved prompt dict only when needed

Preserve the current resolved prompt behavior for:

- `enum`
- scalar root defaults like width/height/frame_count
- `resolved_loras`

- [ ] **Step 4: Preserve plain T2V semantics**

Ensure the resolver leaves `image_path` absent when:

- no root pool exists and the subset has no explicit image
- or the subset sets `use_image_pool = false`

This is important because downstream samplers already interpret missing `image_path` as T2V.

- [ ] **Step 5: Keep failure messages actionable**

Raise `ValueError` or `FileNotFoundError` with messages that tell the operator what to do next, for example:

```python
raise ValueError(
    "Prompt file image pool under [prompt] resolved 3 images, but 5 subsets require pool assignment. "
    "Add more images, mark some subsets with use_image_pool = false, or provide explicit image_path values."
)
```

- [ ] **Step 6: Run resolver tests again to verify green**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest -v tests.test_ltx2_prompt_lora_utils
```

Expected:

- all resolver tests pass

- [ ] **Step 7: Commit the resolver implementation**

```bash
git add src/musubi_tuner/ltx2_prompt_lora_utils.py tests/test_ltx2_prompt_lora_utils.py
git commit -m "feat: add ltx prompt-file image pool resolution"
```

---

## Chunk 3: Verify Integration Paths and Update Docs

### Task 3: Cover sample-prompt precache and user-facing docs

**Files:**
- Modify: `tests/test_ltx2_cache_text_encoder_outputs.py`
- Modify: `docs/ltx_2.md`
- Modify: `docs/config_examples/ltx2_sample_prompts_two_stage.toml`

- [ ] **Step 1: Add a precache integration test**

Extend `tests/test_ltx2_cache_text_encoder_outputs.py` with a temporary prompt TOML and temporary image directory. Verify `_load_sample_prompts_for_precache(...)` returns resolved prompts where:

- pool-assigned subsets have ordinary `image_path`
- `use_image_pool = false` subsets have no `image_path`

Use a test shape like:

```python
def test_load_sample_prompts_for_precache_resolves_prompt_file_image_pool(self):
    prompts = cache_script._load_sample_prompts_for_precache(args)
    self.assertIn("image_path", prompts[0])
    self.assertNotIn("image_path", prompts[1])
```

- [ ] **Step 2: Run the focused cache-precache test**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest -v tests.test_ltx2_cache_text_encoder_outputs
```

Expected:

- all tests pass

- [ ] **Step 3: Update `docs/ltx_2.md`**

Document:

- root `input_images` and `input_images_dir`
- `image_input_order = "sorted" | "random"`
- `image_assignment = "unique"` as the only supported mode in v1
- subset `image_path`
- subset `use_image_pool = false` as the explicit T2V opt-out
- failure behavior when there are too few images

- [ ] **Step 4: Update the example TOML**

Edit `docs/config_examples/ltx2_sample_prompts_two_stage.toml` to include:

- a root `input_images_dir`
- `image_input_order = "random"`
- one subset using inherited pool assignment
- one subset with explicit `image_path`
- one subset with `use_image_pool = false`

Keep the existing two-stage and prompt-local LoRA example value intact where possible.

- [ ] **Step 5: Run the focused end-to-end verification set**

Run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest -v \
  tests.test_ltx2_prompt_lora_utils \
  tests.test_ltx2_cache_text_encoder_outputs
```

Expected:

- all tests pass

- [ ] **Step 6: Commit docs and integration coverage**

```bash
git add tests/test_ltx2_cache_text_encoder_outputs.py docs/ltx_2.md docs/config_examples/ltx2_sample_prompts_two_stage.toml
git commit -m "docs: add ltx prompt-file image pool usage"
```

---

## Final Verification

- [ ] **Step 1: Run the final focused verification command**

```bash
source .venv/bin/activate && PYTHONPATH=src python -m unittest -v \
  tests.test_ltx2_prompt_lora_utils \
  tests.test_ltx2_cache_text_encoder_outputs
```

Expected:

- full command exits `0`
- all targeted tests pass

- [ ] **Step 2: Inspect the final diff**

```bash
git diff --stat HEAD~3..HEAD
```

Expected:

- only resolver, tests, and docs/example files relevant to this feature changed

---

## Notes for the Implementer

- Do not teach the sampler about `input_images_dir`, `image_input_order`, `image_assignment`, or `use_image_pool`. The resolver should lower all of that to final prompt dicts.
- Keep YAGNI strong: v1 supports only `image_assignment = "unique"`.
- Do not add prompt × image expansion, reuse, or cycling behavior in this plan.
- Prefer deterministic tests. If a test mentions `"random"`, control the randomness with patching rather than accepting flaky order.

## Self-Review

This plan stays within one subsystem: the LTX prompt-file resolver and its direct documentation/tests. It intentionally avoids sampler refactors because the spec’s architecture relies on resolving back to the existing `image_path` runtime contract.
