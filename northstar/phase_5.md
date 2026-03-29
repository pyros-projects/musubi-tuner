# Phase 5

## Goal

Add optional ecosystem interop without giving away the hot path or destabilizing the core runtime.

## Why This Phase Was Too Big

PEFT interop, Diffusers config translation, and conversion tooling are related, but they should not ship as one indivisible block.

## Execution Slices

### Phase 5A: Interop Scope and Metadata Contract

**Deliverable**

A clearly defined interop contract for what Northstar will and will not support.

**Scope**

- define supported PEFT adapter shapes
- define supported metadata mapping
- define non-goals
- document unsupported cases explicitly

**Verification**

- written interop contract exists
- tests can target a concrete supported subset

### Phase 5B: PEFT LoRA Import/Export

**Deliverable**

Northstar can export and import the supported LoRA subset.

**Scope**

- add export tool
- add import tool
- add metadata model
- add roundtrip tests

**Verification**

- export a supported LoRA
- import it back
- roundtrip test passes

### Phase 5C: Diffusers-Friendly Config Translation

**Deliverable**

Northstar can translate the supported config subset into a Diffusers-friendly representation.

**Scope**

- add config translation helper
- validate only the cases that map cleanly
- avoid overclaiming compatibility

**Verification**

- translate a simple supported config
- smoke-validate the translated representation where practical

## Exit Criteria

- Interop lands in small honest pieces.
- The runtime remains Northstar-owned and performance-sensitive paths stay local.
