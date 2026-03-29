# Phase 6

## Goal

Turn Northstar from an extraction project into a practical replacement for the targeted `Flux2` and `Z-Image` workflows.

## Why This Phase Was Too Big

Example configs, migration docs, cutover policy, and release readiness are separate pieces of productization work. They should be staged.

## Execution Slices

### Phase 6A: Example Config Pack

**Deliverable**

A tested set of example `my_run.toml` files for the common Northstar workflows.

**Scope**

- publish `Flux2` examples
- publish `Z-Image` examples
- include compile/prewarm examples
- include sampling LoRA examples

**Verification**

- each example starts successfully
- each example is labeled with intended use

### Phase 6B: Migration Guide and Support Matrix

**Deliverable**

A practical guide for moving from Musubi workflows to Northstar workflows.

**Scope**

- map CLI flags to TOML fields
- map dataset TOMLs to inline dataset sections
- map prompt files to `sampling.prompts`
- map cache scripts to cache policy
- define support matrix for first release candidate

**Verification**

- migrate one real Musubi workflow end-to-end
- support matrix is explicit and reviewable

### Phase 6C: Cutover Decision and Release Candidate

**Deliverable**

A clear first alpha or beta cutover story for Northstar.

**Scope**

- decide whether Musubi-backed mode stays:
  - fallback
  - migration bridge
  - deprecated mode
- define release checklist
- run final smoke matrix
- publish release notes and remaining gaps

**Verification**

- final smoke matrix passes for the supported scope
- release checklist is complete
- cutover decision is documented

## Exit Criteria

- Northstar has a usable adoption path.
- Productization work is split into three manageable passes rather than one final giant phase.
