# Compile Prewarm Notes

This document captures ideas for a possible `torch.compile` prewarm flow during training.

## Background

In the current training flow, `--compile` installs `torch.compile(...)` wrappers during startup, but the actual JIT work still happens lazily on the first forward pass for each new input shape / graph signature.

That means:

- early training steps may pause while a new shape is compiled
- bucketed training may trigger multiple separate compiles
- later iterations on already-seen shapes are typically faster

This behavior is already documented in [torch_compile.md](./torch_compile.md): if `--compile_dynamic true` is not used, recompilation can happen whenever input shape changes.

## Goal

The idea of "compile prewarm" is to frontload as much of that JIT cost as possible before the main training loop starts, so epoch 1 has fewer surprise pauses.

Important caveat:

- this can reduce compile pauses
- it cannot guarantee that *all* future recompiles are eliminated

## Why It Cannot Be Perfect

Even with a prewarm phase, later recompiles may still happen because:

- a shape appears during real training that was not included in prewarm
- `torch.compile` decides a new guarded graph is needed
- backward graphs may differ from the warmup graph if warmup is too simplified
- the compile cache may evict older variants
- the model may see different control-flow or tensor metadata than the warmup path

So prewarm should be treated as a best-effort optimization, not a strict guarantee.

## Most Practical First Step

Before building a dedicated prewarm system, the simplest knob to try is:

- `--compile_dynamic true`

This may reduce recompilation across varying bucket shapes, at the cost of potentially lower peak optimization in some cases.

## Proposed Prewarm Strategy

If a dedicated feature is added, the safest version would be a training-style prewarm, not an inference-only dry run.

Suggested flow:

1. Build and prepare the model normally.
2. Enable `torch.compile` as usual.
3. Enumerate the distinct bucket shapes expected during training.
4. For each bucket shape, run one synthetic forward and backward pass that matches the real training path as closely as possible.
5. Skip optimizer stepping and parameter updates.
6. Clear temporary tensors and start normal training.

Why training-style prewarm matters:

- forward-only warmup may leave backward compilation to happen later
- simplified inputs may miss shape-sensitive branches used by real training
- using the real training call path is more likely to compile the graphs that actually matter

## What Would Need To Be Warmed

For best effect, prewarm should match the real training step closely enough to cover:

- latent spatial shape for each resolution bucket
- text / conditioning tensor shapes that vary with the batch
- gradient-enabled training forward
- backward pass
- any architecture-specific branches that depend on shape

For bucketed image/video training, the most important axis is usually the latent shape coming from each bucket.

## Risks and Tradeoffs

Potential benefits:

- smoother start of epoch 1
- fewer visible pauses when new buckets first appear
- easier benchmarking because compile cost is concentrated up front

Potential downsides:

- longer startup time before the first real step
- extra implementation complexity
- more VRAM / RAM pressure during warmup
- no guarantee of eliminating all later recompiles
- may need special handling for block swapping, gradient checkpointing, or architecture-specific code paths

## Cache Considerations

`--compile_cache_size_limit` also matters.

If the number of compiled variants exceeds the cache size, older compiled graphs may be evicted and later recompiled. A prewarm phase can still help, but cache churn may reduce its effect.

## Possible UX

If implemented, a CLI option could look like:

```bash
--compile_prewarm
```

Possible follow-up options:

```bash
--compile_prewarm_max_buckets <N>
--compile_prewarm_forward_only
```

However, the default and recommended version should likely be full training-style prewarm, because forward-only warmup is less reliable.

## Recommended Implementation Order

If this feature is pursued later, a good order would be:

1. Try `--compile_dynamic true` first and measure.
2. Add instrumentation to log when recompiles are likely occurring.
3. Add a minimal prewarm pass for bucket shapes.
4. Expand prewarm to match the real training path more closely if needed.

## Summary

Yes, it is possible to frontload a significant amount of `torch.compile` JIT work.

No, it is not realistic to guarantee that no later compile will ever happen.

The most practical future implementation is:

- keep normal `--compile`
- optionally prewarm per bucket shape using a training-style synthetic step
- treat it as a way to reduce pauses, not as a perfect compile lock-in
