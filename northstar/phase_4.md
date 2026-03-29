# Phase 4

## Goal

Harden Northstar into a credible daily-driver alpha for short experiments and repeated local runs.

## Why This Phase Was Too Big

Benchmarks, resume reliability, parity tooling, and migration docs are four different efforts. They should not be treated as one implementation pass.

## Execution Slices

### Phase 4A: Benchmark Harness

**Deliverable**

A repeatable benchmark harness for short `Flux2` and `Z-Image` runs.

**Scope**

- add benchmark configs
- measure:
  - eager
  - compile
  - compile + prewarm
  - FP8 variants where supported
- record startup and steady-state timings

**Verification**

- benchmark report can be generated locally
- repeated runs produce comparable output structure

### Phase 4B: Checkpoint and Resume Reliability

**Deliverable**

Northstar can save and resume short runs reliably.

**Scope**

- test checkpoint writing
- test checkpoint loading
- verify run metadata survives resume
- verify interrupted short runs can continue

**Verification**

- save/resume smoke test for `Flux2`
- save/resume smoke test for `Z-Image`

### Phase 4C: Parity and Regression Reporting

**Deliverable**

Northstar can produce a parity or regression summary against Musubi reference runs.

**Scope**

- add sample comparison utilities or reports
- add timing comparison summaries
- make drift visible in a compact report

**Verification**

- generate one parity report for `Flux2`
- generate one parity report for `Z-Image`
- report clearly identifies known differences

### Phase 4D: Operator Hardening and Migration Notes

**Deliverable**

Northstar has better user-facing errors and a first migration guide for real experimentation.

**Scope**

- improve validation errors
- improve cache/model-path failure messages
- write first migration notes from Musubi flags to TOML
- publish known gaps

**Verification**

- invalid config errors point to the right TOML keys
- known-gaps doc exists
- migration note covers at least one real Musubi workflow

## Exit Criteria

- Northstar is measured, resumable, and understandable enough for repeated local experiments.
- The hardening work can be implemented in four contained passes instead of one grab-bag phase.
