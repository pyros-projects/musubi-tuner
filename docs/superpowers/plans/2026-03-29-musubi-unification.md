# Musubi Unification Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify `/home/pyro/repos/musubi-tuner` and `/home/pyro/repos/ltx-musubi/musubi-tuner` into a single primary repo by keeping the LTX repo as the base, porting the normal-repo code changes into it, and replacing the OpenSpec process layer with a simple repo-level `AGENTS.md`.

**Architecture:** Treat `/home/pyro/repos/ltx-musubi/musubi-tuner` as the source-of-truth repo because it already contains the full LTX surface area plus recent local fixes. Migrate the normal repo in layers: process/docs first, then the standalone `northstar` package, then shared training/runtime changes, with manual reconciliation only where the two branches overlap.

**Tech Stack:** Git cherry-pick, manual patch reconciliation, Python test suite, existing Musubi trainer entrypoints, repo-level `AGENTS.md` guidance.

---

## File Map

**Target repo (keep):**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner`

**Source repo (port from):**
- Read from: `/home/pyro/repos/musubi-tuner`

**Known overlap files requiring manual reconciliation:**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/hv_train_network.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/utils/model_utils.py`

**New repo guidance file to create in the target repo:**
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/AGENTS.md`

**Normal-repo-only file groups that should port cleanly with low conflict risk:**
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/docs/compile_prewarm.md`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/docs/advanced_config.md`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/docs/flux_2.md`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/docs/torch_compile.md`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/docs/zimage.md`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/examples/flux2_smoke.toml`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/examples/zimage_smoke.toml`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/northstar_framework_extraction_plan.md`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/phase_1.md`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/phase_2.md`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/phase_3.md`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/phase_4.md`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/phase_5.md`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/phase_6.md`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/__init__.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/cli/__init__.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/cli/train.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/config/__init__.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/config/loader.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/config/models.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/core/__init__.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/interop/__init__.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_northstar_cli.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_northstar_config_loader.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_northstar_examples.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_northstar_package.py`

**Shared runtime/training files likely to conflict and require file-by-file review:**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/.gitignore`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/dataset/image_video_dataset.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/hv_train.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/hv_train_network.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/networks/lora.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/networks/lora_flux_2.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/utils/model_utils.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/zimage_train_network.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_compile_prewarm.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_sampling_adapter_compile.py`

---

## Proposed AGENTS.md Draft

```md
# AGENTS.md

## Project Context

- This repository is the primary Musubi working tree.
- `src/musubi_tuner/` is the production reference implementation.
- `src/northstar/` is the clean config-first framework under extraction.
- `northstar/` contains planning docs and example configs, not the Python package itself.

## Primary Goals

- Extract reusable framework pieces without losing Musubi's proven runtime behavior.
- Keep Flux2, Z-Image, and LTX working while Northstar matures.
- Prefer one self-contained run config and a simple launch shape such as `accelerate launch train.py my_run.toml`.

## Architecture Rules

- Prefer extraction over rewrite.
- Keep Musubi behavior as the parity reference until native replacements are validated.
- Avoid premature abstraction.
- Only generalize behavior after at least two families justify the abstraction.
- Make deviations explicit. If code, docs, or plans drift, update the relevant artifact instead of leaving stale guidance behind.

## Performance Rules

- Performance-sensitive behavior is first-class, not cleanup work.
- Be careful around:
  - compile and compile prewarm
  - FP8 and quantization behavior
  - sampling-time LoRA handling
  - cache orchestration and cache invalidation
  - offload, swap, and low-VRAM behavior
- Do not simplify performance-sensitive paths unless validation proves parity.

## Validation Rules

- Every meaningful change should end in a runnable or testable state.
- For behavior-sensitive changes, prefer parity checks against the existing Musubi path.
- Unknowns, trade-offs, and fallback behavior should be made explicit.
- When a change touches training or sampling behavior, add or run focused tests before claiming success.

## Working Conventions

- Keep changes small and reviewable.
- Separate generic framework work from model-family-specific work whenever possible.
- Do not silently expand scope during implementation; update the plan or notes when new required work is discovered.
- Record high-value learnings that save future investigation time.

## High-Value Learnings

- `src/northstar/` is the runtime package location; top-level `northstar/` is for docs and examples.
- The fastest useful Northstar verification slice is:
  - `source .venv/bin/activate && PYTHONPATH=src python -m unittest discover -s tests -p 'test_northstar*.py' -v`
- Prompt numeric fields should be treated as required when a sampling prompt is declared.
- Dataset group `batch_size` should be required, not silently defaulted.
- Config validation should fail cleanly for schema errors, missing files, and TOML parse errors.
- Compile, prewarm, and sampling-adapter behavior are easy places for regressions; test them deliberately.
```

---

## Chunk 1: Repo Baseline And Safety Rails

### Task 1: Create The Unification Branch In The LTX Repo

**Files:**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner`
- Reference: `/home/pyro/repos/musubi-tuner`

- [ ] **Step 1: Confirm both repos are in the expected starting state**

Run:
```bash
git -C /home/pyro/repos/ltx-musubi/musubi-tuner status --short
git -C /home/pyro/repos/ltx-musubi/musubi-tuner branch --show-current
git -C /home/pyro/repos/musubi-tuner status --short
git -C /home/pyro/repos/musubi-tuner branch --show-current
```

Expected:
- LTX repo is on `ltx-2`
- Normal repo is on `feat/northstar`
- Existing dirty files are understood and left alone

- [ ] **Step 2: Create a dedicated integration branch**

Run:
```bash
git -C /home/pyro/repos/ltx-musubi/musubi-tuner switch -c chore/unify-musubi
```

Expected:
- New branch created from current LTX repo HEAD

- [ ] **Step 3: Capture the source commit list inside the branch notes**

Run:
```bash
git -C /home/pyro/repos/musubi-tuner log --reverse --oneline origin/HEAD..HEAD
```

Expected:
- Commit list available for grouping into cherry-pick batches

- [ ] **Step 4: Commit the plan file if needed before execution starts**

Run:
```bash
git -C /home/pyro/repos/ltx-musubi/musubi-tuner add docs/superpowers/plans/2026-03-29-musubi-unification.md
git -C /home/pyro/repos/ltx-musubi/musubi-tuner commit -m "docs: add musubi unification plan"
```

Expected:
- Execution branch starts from a known documented plan

### Task 2: Define The Migration Rule Set

**Files:**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/docs/superpowers/plans/2026-03-29-musubi-unification.md`

- [ ] **Step 1: Lock the base-repo decision**

Rule:
- Keep `/home/pyro/repos/ltx-musubi/musubi-tuner` as the unified repo
- Do not attempt to port the LTX subsystem into `/home/pyro/repos/musubi-tuner`

- [ ] **Step 2: Lock the scope filter**

Rule:
- Port committed code, tests, docs, and selected high-value learnings from the normal repo
- Do not port local-only clutter such as `/home/pyro/repos/musubi-tuner/SageAttention`, `/home/pyro/repos/musubi-tuner/build`, `/home/pyro/repos/musubi-tuner/out`, wheel files, screenshots, or ad hoc local assets
- Do not port the OpenSpec machinery, prompts, or workflow wrappers as-is

- [ ] **Step 3: Lock the conflict strategy**

Rule:
- Cherry-pick clean file groups where possible
- Manually reconcile only shared/core files
- Never overwrite the LTX repo’s `ltx2_*` or `ltx_2/**` code to make unrelated cherry-picks fit

---

## Chunk 2: Establish Lightweight Repo Guidance First

### Task 3: Create AGENTS.md From OpenSpec Learnings And Rules

**Files:**
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/AGENTS.md`
- Reference: `/home/pyro/repos/musubi-tuner/openspec/config.yaml`
- Reference: `/home/pyro/repos/musubi-tuner/openspec/learnings.md`

- [ ] **Step 1: Create `AGENTS.md` using the draft in this plan**

Implementation:
- Copy the draft above into `/home/pyro/repos/ltx-musubi/musubi-tuner/AGENTS.md`
- Keep only durable repo guidance and high-value learnings
- Exclude OpenSpec-specific workflow machinery

- [ ] **Step 2: Review the new `AGENTS.md` for brevity and signal**

Checklist:
- Keep project context and repo role split
- Keep parity-first and extraction-over-rewrite principles
- Keep performance-sensitive warning areas
- Keep validation expectations
- Keep only the most useful learnings from `openspec/learnings.md`
- Remove task bookkeeping, decision-log bureaucracy, and review-loop mandates

- [ ] **Step 3: Commit the lightweight repo guidance**

Run:
```bash
git -C /home/pyro/repos/ltx-musubi/musubi-tuner add AGENTS.md
git -C /home/pyro/repos/ltx-musubi/musubi-tuner commit -m "docs: add repo guidance and migration learnings"
```

Expected:
- Repo-level guidance exists without importing OpenSpec itself

### Task 4: Port The Standalone Northstar Package And Planning Docs

**Files:**
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/examples/flux2_smoke.toml`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/examples/zimage_smoke.toml`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/northstar_framework_extraction_plan.md`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/phase_1.md`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/phase_2.md`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/phase_3.md`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/phase_4.md`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/phase_5.md`
- Create/Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/northstar/phase_6.md`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/__init__.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/cli/__init__.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/cli/train.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/config/__init__.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/config/loader.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/config/models.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/core/__init__.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/northstar/interop/__init__.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_northstar_cli.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_northstar_config_loader.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_northstar_examples.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_northstar_package.py`

- [ ] **Step 1: Cherry-pick the Northstar package series in source order**

Run:
```bash
git -C /home/pyro/repos/ltx-musubi/musubi-tuner cherry-pick -n 7499deb 4ba11fb 34d663f 9665ed7 d6f7bf9 ec45d6b 6441c6d cc4890f aa51217
```

Expected:
- `src/northstar/**`, examples, tests, and supporting docs land in the LTX repo

- [ ] **Step 2: Review only the Northstar-specific paths**

Run:
```bash
git -C /home/pyro/repos/ltx-musubi/musubi-tuner diff -- northstar src/northstar tests/test_northstar_cli.py tests/test_northstar_config_loader.py tests/test_northstar_examples.py tests/test_northstar_package.py
```

Expected:
- All files are self-contained and not entangled with LTX internals

- [ ] **Step 3: Run the Northstar-focused tests**

Run:
```bash
cd /home/pyro/repos/ltx-musubi/musubi-tuner && python -m pytest tests/test_northstar_cli.py tests/test_northstar_config_loader.py tests/test_northstar_examples.py tests/test_northstar_package.py -q
```

Expected:
- Northstar package tests pass in the LTX repo

- [ ] **Step 4: Commit the Northstar import**

Run:
```bash
git -C /home/pyro/repos/ltx-musubi/musubi-tuner add northstar src/northstar tests/test_northstar_cli.py tests/test_northstar_config_loader.py tests/test_northstar_examples.py tests/test_northstar_package.py
git -C /home/pyro/repos/ltx-musubi/musubi-tuner commit -m "feat: import northstar package scaffolding"
```

Expected:
- Northstar lands as its own reviewable commit

---

## Chunk 3: Reconcile Shared Training And Runtime Changes

### Task 5: Port Compile-Prewarm And Sampling-Adapter Improvements

**Files:**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/.gitignore`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/docs/compile_prewarm.md`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/docs/flux_2.md`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/docs/torch_compile.md`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/docs/zimage.md`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/hv_train_network.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/utils/model_utils.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_compile_prewarm.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_sampling_adapter_compile.py`

- [ ] **Step 1: Cherry-pick the compile/sampling series without committing**

Run:
```bash
git -C /home/pyro/repos/ltx-musubi/musubi-tuner cherry-pick -n c097f7c cfe4329 44c3337
```

Expected:
- Shared runtime improvements appear in the worktree

- [ ] **Step 2: Manually reconcile `hv_train_network.py`**

Focus:
- Preserve the LTX repo’s current non-LTX and LTX behavior
- Port compile-prewarm and sampling-adapter persistence logic only where it applies generically
- Do not regress the recent LTX-only sampling fixes already on `ltx-2`

- [ ] **Step 3: Manually reconcile `utils/model_utils.py`**

Focus:
- Preserve any LTX-specific helper logic already present
- Port only the generic compile/runtime helper additions from the normal repo

- [ ] **Step 4: Run the compile-related tests**

Run:
```bash
cd /home/pyro/repos/ltx-musubi/musubi-tuner && python -m pytest tests/test_compile_prewarm.py tests/test_sampling_adapter_compile.py -q
```

Expected:
- Compile feature tests pass in the unified repo

- [ ] **Step 5: Commit the compile/runtime integration**

Run:
```bash
git -C /home/pyro/repos/ltx-musubi/musubi-tuner add .gitignore docs/compile_prewarm.md docs/flux_2.md docs/torch_compile.md docs/zimage.md src/musubi_tuner/hv_train_network.py src/musubi_tuner/utils/model_utils.py tests/test_compile_prewarm.py tests/test_sampling_adapter_compile.py
git -C /home/pyro/repos/ltx-musubi/musubi-tuner commit -m "feat: port compile prewarm and sampling adapter improvements"
```

Expected:
- Shared runtime features land in one isolated commit

### Task 6: Port The Remaining Generic Trainer And Validation Fixes

**Files:**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/dataset/image_video_dataset.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/hv_train.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/hv_train_network.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/networks/lora.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/networks/lora_flux_2.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/zimage_train_network.py`

- [ ] **Step 1: Cherry-pick the validation/fix series without committing**

Run:
```bash
git -C /home/pyro/repos/ltx-musubi/musubi-tuner cherry-pick -n 7be9623 814cccd d97d4ab 16836c1 1bc2d1f
```

Expected:
- Generic validation fixes appear in the worktree

- [ ] **Step 2: Review each touched training file for model-family assumptions**

Checklist:
- `hv_train_network.py`: keep family-agnostic validation only
- `image_video_dataset.py`: preserve current dataset semantics in the LTX repo
- `lora.py` and `lora_flux_2.py`: keep generic LoRA behavior, avoid backsliding any LTX-specific conventions
- `zimage_train_network.py`: port only if the target file still matches the source assumptions

- [ ] **Step 3: Run focused trainer smoke tests or unit tests that cover the changed code**

Run:
```bash
cd /home/pyro/repos/ltx-musubi/musubi-tuner && python -m pytest tests/test_compile_prewarm.py tests/test_sampling_adapter_compile.py -q
```

Expected:
- No regression in the already-ported compile/runtime tests

- [ ] **Step 4: Commit the generic trainer fixes**

Run:
```bash
git -C /home/pyro/repos/ltx-musubi/musubi-tuner add src/musubi_tuner/dataset/image_video_dataset.py src/musubi_tuner/hv_train.py src/musubi_tuner/hv_train_network.py src/musubi_tuner/networks/lora.py src/musubi_tuner/networks/lora_flux_2.py src/musubi_tuner/zimage_train_network.py
git -C /home/pyro/repos/ltx-musubi/musubi-tuner commit -m "fix: port generic trainer validation improvements"
```

Expected:
- Remaining generic fixes land separately from the compile integration

---

## Chunk 4: Final Verification And Cutover

### Task 7: Run The Unified Verification Pass

**Files:**
- Verify: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests`

- [ ] **Step 1: Run the imported and existing focused tests together**

Run:
```bash
cd /home/pyro/repos/ltx-musubi/musubi-tuner && python -m pytest tests/test_northstar_cli.py tests/test_northstar_config_loader.py tests/test_northstar_examples.py tests/test_northstar_package.py tests/test_compile_prewarm.py tests/test_sampling_adapter_compile.py tests/test_ltx2_sampling_lora.py -q
```

Expected:
- Imported Northstar tests pass
- Shared compile/runtime tests pass
- Existing LTX sampling tests still pass

- [ ] **Step 2: Run a static scan for accidental source-repo leftovers**

Run:
```bash
git -C /home/pyro/repos/ltx-musubi/musubi-tuner diff --name-only origin/ltx-2..HEAD
```

Expected:
- Only intentional unified-repo files appear

- [ ] **Step 3: Review the final change summary**

Run:
```bash
git -C /home/pyro/repos/ltx-musubi/musubi-tuner log --oneline --decorate origin/ltx-2..HEAD
git -C /home/pyro/repos/ltx-musubi/musubi-tuner diff --stat origin/ltx-2..HEAD
```

Expected:
- Final commit stack is understandable and grouped by concern

### Task 8: Cut Over To One Primary Repo

**Files:**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner`
- Archive later: `/home/pyro/repos/musubi-tuner`

- [ ] **Step 1: Push the unified branch to your fork**

Run:
```bash
git -C /home/pyro/repos/ltx-musubi/musubi-tuner push pyros-projects chore/unify-musubi
```

Expected:
- Unified branch is safely backed up remotely

- [ ] **Step 2: Decide the final local repo role**

Rule:
- Keep `/home/pyro/repos/ltx-musubi/musubi-tuner` as the active working repo
- Freeze `/home/pyro/repos/musubi-tuner` as reference-only until confidence is high

- [ ] **Step 3: Document the cutover in the repo root or team notes**

Suggested note:
- “Primary musubi repo is now `/home/pyro/repos/ltx-musubi/musubi-tuner`; old `/home/pyro/repos/musubi-tuner` is read-only reference pending retirement.”

---

## Commit Grouping Summary

Use this commit order from the normal repo as the import backbone:

1. Manual `AGENTS.md` creation using extracted rules and learnings from `openspec/config.yaml` and `openspec/learnings.md`
2. `7499deb 4ba11fb 34d663f 9665ed7 d6f7bf9 ec45d6b 6441c6d cc4890f aa51217` for Northstar scaffolding and validation
3. `c097f7c cfe4329 44c3337` for compile prewarm and sampling adapter behavior
4. `7be9623 814cccd d97d4ab 16836c1 1bc2d1f` for remaining generic validation fixes

Do **not** blindly cherry-pick everything as one batch. The only known shared-code overlap is:
- `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/hv_train_network.py`
- `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/utils/model_utils.py`

Those two files should be handled deliberately after each cherry-pick batch.

---

Plan complete and saved to `docs/superpowers/plans/2026-03-29-musubi-unification.md`. Ready to execute?
