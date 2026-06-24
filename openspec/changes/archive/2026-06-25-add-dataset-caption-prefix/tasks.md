## 1. Dataset Config Plumbing

- [x] 1.1 Add `caption_prefix: str = ""` to shared dataset parameter dataclasses and validation schemas.
- [x] 1.2 Ensure `[general] caption_prefix` falls back into each dataset and per-dataset `caption_prefix` overrides it.
- [x] 1.3 Include `caption_prefix` in dataset logging/metadata surfaces where caption-related dataset settings are printed or serialized.

## 2. Caption Materialization

- [x] 2.1 Add a shared helper for exact raw prefix concatenation.
- [x] 2.2 Pass `caption_prefix` into image, video, and audio directory datasources.
- [x] 2.3 Pass `caption_prefix` into image, video, and audio JSONL datasources.
- [x] 2.4 Apply the prefix before captions become `ItemInfo.caption` for latent cache, text cache, and training paths.
- [x] 2.5 Allow directory-backed datasets with non-empty `caption_prefix` to use the prefix itself when an item sidecar is missing.
- [x] 2.6 Preserve existing missing-sidecar strictness when `caption_prefix` is empty.
- [x] 2.7 Handle `multiple_target=true` image datasets clearly so prefix fallback does not make base/target grouping ambiguous.

## 3. Text Cache Safety

- [x] 3.1 Update text encoder cache `--skip_existing` filtering to compare existing `caption1` metadata with the current effective caption.
- [x] 3.2 Re-encode cache files when `caption1` is missing or mismatched.
- [x] 3.3 Preserve current cache cleanup behavior for cache files not in the dataset.

## 4. Documentation And GUI Export

- [x] 4.1 Document `caption_prefix` in `docs/dataset_config.md`, including global/per-dataset usage and raw concatenation semantics.
- [x] 4.2 Add `caption_prefix` to GUI dataset schema models.
- [x] 4.3 Add `caption_prefix` to GUI TOML preview/export code for training and validation datasets.
- [x] 4.4 Add a dataset editor field for `caption_prefix` in the GUI frontend source.
- [x] 4.5 Rebuild committed GUI frontend artifacts only if this repo expects built bundles to be committed for GUI changes.

## 5. Tests And Validation

- [x] 5.1 Add CPU tests for global fallback, per-dataset override, and omitted-prefix default behavior.
- [x] 5.2 Add CPU tests for sidecar caption prefixing, exact concatenation, empty sidecar prefixing, and missing-sidecar prefix fallback.
- [x] 5.3 Add CPU tests for JSONL caption prefixing across representative datasource types.
- [x] 5.4 Add CPU tests for text cache skip behavior with matching, mismatched, and missing `caption1` metadata.
- [x] 5.5 Run focused tests for the new dataset prefix behavior.
- [x] 5.6 Run `openspec validate add-dataset-caption-prefix --strict`.
