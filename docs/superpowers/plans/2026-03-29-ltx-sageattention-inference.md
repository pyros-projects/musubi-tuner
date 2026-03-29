# LTX SageAttention Inference Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SageAttention support to LTX for standalone inference and preview sampling during training, without attempting full training-path support.

**Architecture:** Extend LTX’s existing attention-backend plumbing so `sageattn` becomes a first-class backend alongside `torch`, `flash`, `flash3`, and `xformers`. Keep the scope limited to inference-facing paths by adding backend selection, runtime availability checks, safe fallback/error behavior, and focused verification for standalone generation and training preview sampling.

**Tech Stack:** Python, PyTorch, LTX custom transformer attention backend selection, SageAttention Python package, argparse, focused unit/integration-style tests.

---

## Context Summary

Current LTX attention support is limited to:
- `torch` / `sdpa`
- `flash` / `flash2`
- `flash3`
- `xformers`

This is visible in:
- [ltx2_generate_video.py](/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/ltx2_generate_video.py#L118)
- [ltx2_train_network.py](/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/ltx2_train_network.py#L2447)
- [attention.py](/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/ltx_2/model/transformer/attention.py#L214)

LTX attention runtime currently centralizes backend behavior in:
- [attention.py](/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/ltx_2/model/transformer/attention.py)

That is the right integration point for SageAttention.

There is also a valuable external reference implementation in:
- [ltxv_nodes.py](/home/pyro/repos/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py#L1608)

Important observation from that reference:
- KJNodes patches only `attn1` on each LTX transformer block, via [ltxv_nodes.py](/home/pyro/repos/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py#L1636)
- the custom forward path is implemented in [ltxv_nodes.py](/home/pyro/repos/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py#L1688)
- it uses SageAttention core kernels directly and assumes a no-mask path

This is strong evidence that:
- LTX self-attention inference is feasible with SageAttention
- the RoPE + Q/K/V layout problem is likely solvable

This does **not** prove that:
- masked cross-attention is supported
- audio/video cross-attention is supported
- a generic “all LTX attention uses SageAttention” backend is safe without fallback logic

This plan intentionally excludes:
- full training forward/backward support
- claims about gradient correctness under SageAttention
- retrofitting non-LTX families
- aggressive performance tuning beyond “works correctly and predictably”

---

## Effort And Risk

**Effort estimate**

- Best case: `0.5 day`
  - if we scope the first implementation to the already-proven self-attention path and explicitly fall back elsewhere
- Expected case: `1-2 days`
  - if we integrate the KJNodes-style self-attention path cleanly into Musubi’s backend plumbing, plus fallback logic and tests
- High-friction case: `2-4 days`
  - if SageAttention does not support one of LTX’s required shapes, dtypes, or attention mask patterns cleanly

**Risk level:** Medium

**Why it is not low risk**

- LTX attention is not a thin wrapper over SDPA; it supports:
  - self-attention
  - cross-attention
  - AV cross-attention
  - optional masks
  - split-attention chunking
- The main uncertainty is whether SageAttention supports the exact Q/K/V layout and masking behavior used by LTX.

**Main technical risks**

1. SageAttention may not support the masked cross-attention cases LTX needs.
2. LTX split-attention paths may interact awkwardly with SageAttention’s expected tensor format.
3. Some inference configurations may run in dtypes or devices that SageAttention rejects.
4. Preview sampling uses the same model path but different runtime conditions, so backend availability/fallback behavior must be consistent across both entrypoints.

**Risk reduction from KJNodes reference**

- We no longer have to guess whether plain LTX self-attention can work with SageAttention.
- We still do have to decide whether the Musubi feature contract should be:
  - “SageAttention accelerates self-attention where supported and falls back elsewhere”, or
  - “SageAttention is only exposed when the whole requested path is supported”

**Main product risk**

- A half-working `sageattn` flag would be worse than no support if it silently falls back unpredictably or only works for some prompt/sample paths.

---

## Recommended Approach

### Option A: First-class `sageattn` backend with explicit self-attention-first support

Add:
- parser support for `sageattn`
- a new `SageAttention` backend adapter in `attention.py`
- model-construction plumbing so LTX attention modules receive that backend
- use the KJNodes self-attention implementation as a technical reference
- focused tests and runtime validation

**Pros**
- Cleanest UX
- Reuses existing backend-selection architecture
- Works for both standalone inference and preview sampling through the same code path
- Already has a known-good self-attention reference to borrow from

**Cons**
- Requires careful adapter work for masks and layouts

### Option B: Best-effort inference-only monkey patch outside attention.py

Patch LTX attention modules after model load in inference code only.

**Pros**
- Smaller initial diff

**Cons**
- More fragile
- Duplicates logic across standalone inference and training preview paths
- Harder to reason about and test

### Option C: Add parser support only and hard-fallback to torch when unsupported

Recognize `sageattn` in args, but only use it for trivial cases and fall back elsewhere.

**Pros**
- Fastest to prototype

**Cons**
- Confusing UX
- Easy to create “it says SageAttention but didn’t actually use it” behavior

**Recommendation:** Option A

It fits the existing architecture best and gives the cleanest long-term behavior. The implementation is still relatively contained because LTX backend dispatch is centralized in one module, and KJNodes gives us a concrete reference for the most important subproblem: self-attention inference.

---

## File Map

**Primary implementation files**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/ltx_2/model/transformer/attention.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/ltx2_generate_video.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/ltx2_train_network.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/docs/ltx_2.md`

**Likely test files**
- Modify or Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_ltx2_sampling_lora.py`
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_ltx2_sageattention.py`

**Potential helper file if adapter logic gets too large**
- Create: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/ltx_2/model/transformer/sageattention_adapter.py`

---

## Implementation Strategy

## Chunk 1: Feasibility And API Fit

### Task 1: Confirm SageAttention Callable Surface Against LTX Tensor Shapes

**Files:**
- Inspect: `/home/pyro/repos/ltx-musubi/musubi-tuner/.venv/lib/python3.10/site-packages/sageattention`
- Inspect: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/ltx_2/model/transformer/attention.py`
- Test: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_ltx2_sageattention.py`

- [ ] **Step 1: Inspect the installed SageAttention API**

Run:
```bash
cd /home/pyro/repos/ltx-musubi/musubi-tuner && .venv/bin/python - <<'PY'
import inspect
import sageattention
print(sageattention)
print(dir(sageattention))
PY
```

Expected:
- Identify the callable(s) intended for runtime attention execution

- [ ] **Step 2: Map LTX tensor layout to SageAttention expectations**

Checklist:
- LTX reshapes to `[B, M, H, K]` for xformers and flash-style backends
- KJNodes reshapes LTX attention exactly this way in [ltxv_nodes.py](/home/pyro/repos/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py#L1709)
- Verify what SageAttention expects
- Verify whether it supports:
  - BF16 / FP16 inference
  - causal=`False`
  - non-causal cross-attention
  - optional masks or padding masks

- [ ] **Step 3: Write a tiny failing test or probe for backend invocation**

Implementation idea:
- create a minimal test that imports the adapter and runs a tiny random attention call on CUDA when SageAttention is installed

- [ ] **Step 4: Decide the mask policy**

Decision to make explicitly:
- If SageAttention supports required masks, use it for all inference cases
- If it does not, either:
  - hard-error when `sageattn` is requested in unsupported masked paths, or
  - fall back to PyTorch with an explicit warning

**Recommended:** explicit warning + fallback for unsupported masked cases

- [ ] **Step 5: Compare against the KJNodes patch explicitly**

Files:
- Inspect: `/home/pyro/repos/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py`

Checklist:
- identify which SageAttention kernels KJNodes uses per architecture
- identify which parts are Comfy-specific and which are portable
- identify whether fused RoPE handling is worth porting or should remain separate
- confirm that KJNodes does not attempt masked or cross-attention coverage

### Task 2: Freeze The Product Contract

**Files:**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/docs/ltx_2.md`
- Test: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_ltx2_sageattention.py`

- [ ] **Step 1: Define the supported CLI contract**

Contract:
- Standalone inference:
  - `--attn_mode sageattn`
- Preview sampling during training:
  - either `--attn_mode sageattn` if training parser already supports it cleanly
  - or a dedicated boolean compatibility flag if parser structure forces that

**Recommended:** use `--attn_mode sageattn` as the canonical interface for LTX

- [ ] **Step 2: Define failure behavior**

Contract:
- If SageAttention is requested but not importable, raise a clear runtime error
- If SageAttention is requested for an unsupported masked case, warn and fall back to PyTorch SDPA
- Do not silently pretend SageAttention is active when it is not

---

## Chunk 2: Backend Plumbing

### Task 3: Add A SageAttention Backend To LTX Attention

**Files:**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/ltx_2/model/transformer/attention.py`
- Test: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_ltx2_sageattention.py`

- [ ] **Step 1: Write the failing backend-selection tests**

Test cases:
- selecting `sageattn` maps to a dedicated backend
- missing library raises a clear error
- unsupported mask path triggers explicit fallback or explicit error, whichever contract was chosen

- [ ] **Step 2: Add import detection for SageAttention**

Implementation:
- mirror the existing optional-import pattern used for xformers and flash-attn
- keep import failure cheap and explicit

- [ ] **Step 3: Add `SageAttention` callable and enum value**

Implementation:
- create a `SageAttention` backend class implementing the `AttentionCallable` protocol
- extend `AttentionFunction` with a SageAttention option
- keep layout conversions localized here
- use the KJNodes forward path as the starting reference for:
  - Q/K normalization order
  - RoPE application timing
  - reshape to `[B, seq, heads, head_dim]`
  - architecture-specific SageAttention kernel choice

- [ ] **Step 4: Handle masks deliberately**

Implementation:
- if SageAttention has the necessary mask support, implement it directly
- otherwise detect masked inputs and route to `PytorchAttention()` with a warning

- [ ] **Step 4.5: Decide whether to limit the first implementation to self-attention**

Recommended MVP:
- enable SageAttention only for unmasked self-attention cases first
- fall back to the existing backend for cross-attention and masked paths

Reason:
- KJNodes proves this path
- it sharply reduces implementation risk
- it still likely captures a meaningful chunk of inference VRAM savings

- [ ] **Step 5: Run the backend tests**

Run:
```bash
cd /home/pyro/repos/ltx-musubi/musubi-tuner && python -m pytest tests/test_ltx2_sageattention.py -q
```

Expected:
- backend selection behavior is covered

### Task 4: Wire `sageattn` Into LTX Model Construction

**Files:**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/ltx2_train_network.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/src/musubi_tuner/ltx2_generate_video.py`
- Test: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_ltx2_sageattention.py`

- [ ] **Step 1: Write failing parser/selection tests**

Test cases:
- `ltx2_generate_video.py` accepts `--attn_mode sageattn`
- training preview path can select SageAttention for sampling

- [ ] **Step 2: Extend standalone inference parser**

Implementation:
- add `sageattn` to the `--attn_mode` choices in `ltx2_generate_video.py`

- [ ] **Step 3: Extend LTX transformer loader mapping**

Implementation:
- update `load_transformer()` in `ltx2_train_network.py` so `sageattn` can be selected instead of being collapsed back to `torch`
- ensure `self._dit_attn_mode` records it correctly for sampling/runtime introspection

- [ ] **Step 4: Verify preview-sampling path uses the same backend**

Implementation:
- trace the same transformer load path used by training-time sampling
- ensure preview sampling honors the selected backend without new special cases

- [ ] **Step 5: Run parser/backend tests**

Run:
```bash
cd /home/pyro/repos/ltx-musubi/musubi-tuner && python -m pytest tests/test_ltx2_sageattention.py -q
```

Expected:
- CLI/backend plumbing works end to end

---

## Chunk 3: Runtime Validation For The Requested Scope

### Task 5: Validate Standalone Inference

**Files:**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_ltx2_sageattention.py`

- [ ] **Step 1: Add a focused inference smoke test or mock-based contract test**

Goal:
- verify that selecting `sageattn` reaches the SageAttention backend in an inference-shaped call

- [ ] **Step 2: If practical, run one tiny real CUDA smoke invocation**

Run example:
```bash
cd /home/pyro/repos/ltx-musubi/musubi-tuner && python -m pytest tests/test_ltx2_sageattention.py -q
```

Expected:
- backend invocation succeeds on the installed CUDA environment

### Task 6: Validate Preview Sampling During Training

**Files:**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_ltx2_sampling_lora.py`
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/tests/test_ltx2_sageattention.py`

- [ ] **Step 1: Add a preview-sampling regression test**

Goal:
- ensure sample-generation code path respects the selected backend
- ensure no regression in prompt caching / audio-disable behavior already fixed on this branch

- [ ] **Step 2: Re-run the existing LTX sampling tests**

Run:
```bash
cd /home/pyro/repos/ltx-musubi/musubi-tuner && python -m unittest tests.test_ltx2_sampling_lora
```

Expected:
- existing preview-sampling behavior still passes

---

## Chunk 4: Docs And Finish Criteria

### Task 7: Document The Backend And Its Caveats

**Files:**
- Modify: `/home/pyro/repos/ltx-musubi/musubi-tuner/docs/ltx_2.md`

- [ ] **Step 1: Document the new backend flag**

Add:
- `--attn_mode sageattn`
- any corresponding config-portability boolean if one was added

- [ ] **Step 2: Document caveats honestly**

Document:
- inference and preview sampling only
- not guaranteed for full training path
- any fallback behavior for masks or unsupported cases
- dependency requirement on installed `sageattention`

- [ ] **Step 3: Commit the docs and implementation**

Run:
```bash
git add src/musubi_tuner/ltx_2/model/transformer/attention.py src/musubi_tuner/ltx2_generate_video.py src/musubi_tuner/ltx2_train_network.py docs/ltx_2.md tests/test_ltx2_sageattention.py tests/test_ltx2_sampling_lora.py
git commit -m "feat: add SageAttention backend for LTX inference"
```

Expected:
- one reviewable feature commit

---

## Success Criteria

- `sageattn` is a supported LTX backend for standalone inference
- training preview sampling can use the same backend
- unsupported cases fail clearly or fall back explicitly
- existing LTX sampling behavior does not regress
- docs clearly state the scope limits
- implementation behavior is consistent with the actual supported scope, even if that scope is “self-attention only with fallback for the rest”

## No-Go Criteria

Do not proceed with implementation if early feasibility checks show any of these:

- SageAttention cannot handle LTX’s required non-causal cross-attention shape
- SageAttention cannot work with the dtypes LTX inference actually uses
- the only viable solution would require patching half the LTX model stack rather than the centralized attention backend

If any no-go criterion is hit, stop after Chunk 1 and write up the findings instead of forcing the feature.

---

## Recommendation

This is worth attempting, but only under the narrow scope you chose.

My honest read:
- parser/docs plumbing is easy
- backend wiring is moderate
- KJNodes materially lowers the self-attention feasibility risk
- mask and cross-attention compatibility are still the real unknowns

So I’d call this:
- **effort:** medium
- **risk:** medium
- **odds of getting a useful scoped version working:** good, especially for a self-attention-first MVP
- **odds of it turning into a clean “just another backend” feature for all LTX paths:** much lower

Plan complete and saved to `docs/superpowers/plans/2026-03-29-ltx-sageattention-inference.md`. Ready to execute?
