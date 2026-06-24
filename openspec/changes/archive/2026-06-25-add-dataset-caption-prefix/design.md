## Context

Musubi dataset configs already have ascendable keys such as `caption_extension`, `batch_size`, and `num_repeats` that can be set in `[general]` or overridden per `[[datasets]]`. Captions enter the runtime through shared datasource classes in `image_video_dataset.py`, then become `ItemInfo.caption`, which is reused by latent caching, text encoder caching, training metadata, and trainer batches.

Diffusion-pipe supports `caption_prefix` by raw-prepending the configured prefix to every caption, and by using the prefix itself when no caption sidecar exists. Musubi currently has no equivalent, and its image-directory path filters out images without sidecars whenever `caption_extension` is set.

## Goals / Non-Goals

**Goals:**
- Add `caption_prefix` as an ascendable dataset config key with `[general]` fallback and per-dataset override.
- Apply the prefix once, at caption materialization time, before `ItemInfo.caption` is created.
- Match diffusion-pipe concatenation semantics: no implicit delimiter or whitespace.
- Allow directory-backed datasets to use the prefix as the full caption when caption sidecars are absent or omitted.
- Prevent `--skip_existing` from silently reusing text encoder caches whose metadata caption no longer matches the effective prefixed caption.
- Keep support generic across image, video, audio, sidecar text, and JSONL-backed datasources.
- Surface the field through docs and GUI TOML export so configs remain round-trippable.

**Non-Goals:**
- No caption suffix support.
- No prompt templating, tag shuffling, trigger-token parsing, or automatic comma/space insertion.
- No model-family-specific behavior for Krea2, LTX, Qwen, Z-Image, or Wan.
- No migration of existing sidecar files or cache directories.

## Decisions

### Add `caption_prefix` to the shared dataset parameter path

`caption_prefix` should be added to `BaseDatasetParams` and `ConfigSanitizer.DATASET_ASCENDABLE_SCHEMA`, then passed through `BaseDataset`, `ImageDataset`, `VideoDataset`, and `AudioDataset`. This mirrors `caption_extension` and avoids adding model-specific arguments.

Alternative considered: add a CLI flag to Krea2 cache/training scripts. That would solve the immediate Krea2 use case but would diverge from Musubi's dataset-config-driven shape and would not help existing LTX/Z-Image/Qwen configs.

### Prefix at datasource caption read boundaries

The datasource classes should receive `caption_prefix` and apply it in `get_caption()` / `get_*_data()` when reading sidecars or JSONL captions. The final value stored in `ItemInfo.caption` should already include the prefix.

Alternative considered: prefix during text encoder caching only. That would leave training metadata/debug paths inconsistent and would fail for flows that consume cached or direct captions outside the text cache script.

### Preserve raw concatenation semantics

The implementation should use `caption_prefix + caption` exactly. Users remain responsible for writing `caption_prefix = "middlesplits "` or `caption_prefix = "middlesplits, "`.

Alternative considered: insert a space when both prefix and caption are non-empty. That is friendlier for casual use, but it diverges from diffusion-pipe and can corrupt carefully authored prompt templates.

### Treat missing sidecars as prefix-only when a prefix exists

For directory-backed datasets, if `caption_prefix` is non-empty and a caption file does not exist, the effective caption should be exactly `caption_prefix`. If `caption_prefix` is empty, existing sidecar expectations should remain unchanged.

Image discovery needs special handling because `glob_images(..., caption_extension=...)` currently filters out images without sidecars. When prefix fallback is enabled, image discovery should keep images even when the sidecar is missing. For `multiple_target=true`, sidecar matching is part of the base-image discovery convention, so implementation should either keep strict sidecar discovery or fail clearly if prefix fallback would make target grouping ambiguous.

Alternative considered: only support prefix fallback when `caption_extension` is omitted. That misses the common mixed case where most items have sidecars but a subset should use the trigger-only prompt.

### Make text cache skipping caption-aware

Text encoder cache filenames are derived from item keys, not caption hashes. Existing `--skip_existing` logic skips purely by path, before reading cache metadata. The skip check should only skip an existing cache if its `caption1` metadata equals the current effective caption. Missing metadata or mismatched captions should force re-encoding.

Alternative considered: include a caption hash in cache filenames. That avoids metadata reads but would be a broader cache layout migration and would leave old caches behind unless additional cleanup logic is added.

### Keep GUI support narrow

The GUI should add `caption_prefix` to its dataset schema, editor field, and TOML export paths. It does not need special validation beyond string handling. The source frontend should be updated; generated frontend bundles should only be updated if this repo's normal GUI build process requires committed bundles.

## Risks / Trade-offs

- **Risk: Existing configs with `caption_prefix = ""` change behavior** -> Keep the default empty string and preserve current sidecar requirements when the prefix is empty.
- **Risk: Prefix fallback includes unintended images in image directories** -> Only relax image sidecar filtering when `caption_prefix` is non-empty; call out `multiple_target=true` as strict/ambiguous.
- **Risk: Metadata-aware `--skip_existing` slows large text-cache scans** -> Only inspect metadata for candidate existing cache files when skip-existing is active; this is cheaper than silently training against stale embeddings.
- **Risk: GUI and CLI config paths drift** -> Update both `project_schema.py`/export code and docs/tests in the same change.
- **Risk: Cache-only manifests contain source-free params where prefix is no longer relevant** -> Preserve `caption_prefix` in metadata/manifests for observability, but training from cache-only manifests continues to consume cached embeddings and latents rather than re-reading source captions.

## Migration Plan

1. Implement the shared config field and caption materialization changes.
2. Update skip-existing text cache behavior so changed prefixes recache embeddings automatically.
3. Update docs and GUI export surfaces.
4. Add focused CPU tests and run them locally.
5. Existing datasets continue to work unchanged because the default prefix is empty.

Rollback is straightforward: remove the config key from TOML or set it to an empty string, then recache text embeddings if prefixed caches were generated.

## Open Questions

- Should `multiple_target=true` with `caption_prefix` and missing sidecars fail immediately, or preserve strict sidecar filtering and ignore fallback for target grouping? Recommended: fail clearly if a user tries to rely on missing-sidecar fallback with `multiple_target=true`.
