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
