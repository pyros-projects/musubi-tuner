# Phase 1

## Goal

Build the first runnable `northstar` shell that owns config loading, run setup, and operator UX while still delegating real execution to Musubi adapters.

## Why This Phase Was Too Big

The original Phase 1 mixed four different deliverables:
- config design
- runtime scaffolding
- Musubi interop
- automatic cache orchestration

That is too much for one clean implementation pass, so this phase is now split into four execution slices.

## Execution Slices

### Phase 1A: Package Skeleton and Minimal CLI

**Deliverable**

A callable `northstar` CLI that can load a tiny TOML with just enough fields to identify a run and print a validated startup summary.

**Scope**

- create the `northstar/` package skeleton
- add `northstar/cli/train.py`
- add minimal config loading for:
  - `run`
  - `model`
- fail cleanly on malformed TOML

**Verification**

- `python -m northstar.cli.train examples/minimal_flux2.toml`
- config validation errors are readable
- startup summary shows run name and architecture

### Phase 1B: Full Single-File Config Contract

**Deliverable**

A strict single-file run config that supports the full section layout, even if execution is still partial.

**Scope**

- add the complete section structure:
  - `run`
  - `accelerate`
  - `model`
  - `training`
  - `optimizer`
  - `network`
  - `performance`
  - `output`
  - `sampling`
  - `dataset`
- fail on unknown keys
- add migration-friendly error messages
- add example `Flux2` and `Z-Image` smoke configs

**Verification**

- validation passes for example configs
- validation fails for misspelled keys
- parsed config can be round-tripped into a stable manifest form

### Phase 1C: Musubi-Backed Execution Adapters

**Deliverable**

Northstar can launch Musubi-backed smoke runs for `Flux2` and `Z-Image` using one TOML file.

**Scope**

- add `Flux2` adapter
- add `Z-Image` adapter
- translate inline prompts into Musubi-compatible sampling input
- translate inline dataset config into Musubi-compatible dataset input
- centralize config-to-Musubi translation logic

**Verification**

- `Flux2` smoke run launches via Northstar
- `Z-Image` smoke run launches via Northstar
- no external prompt file is needed
- no external dataset TOML is needed for the smoke path

### Phase 1D: Automatic Cache Preparation and Run Metadata

**Deliverable**

The Northstar CLI can inspect caches, prepare them when needed, and record what happened.

**Scope**

- add cache policy fields to the run config
- inspect required caches before training
- invoke Musubi cache preparation when needed
- record:
  - cache reused
  - cache created
  - cache rebuilt
- write stable run metadata and config snapshot

**Verification**

- first run creates missing caches
- second run reuses existing caches
- run directory contains manifest and config copy

## Exit Criteria

- One TOML file is enough for a normal Musubi-backed smoke run.
- Cache preparation is part of the Northstar workflow.
- The phase can be executed slice-by-slice without overloading one implementation pass.
