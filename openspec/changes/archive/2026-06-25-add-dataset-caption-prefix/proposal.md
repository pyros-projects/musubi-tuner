## Why

Musubi dataset configs can select caption sidecars or JSONL captions, but they cannot add a stable trigger word or fixed prompt prefix at config time. Pyro currently has this workflow in diffusion-pipe, and Krea2/LTX runs need the same low-friction operator surface without rewriting thousands of sidecar files.

## What Changes

- Add a `caption_prefix` dataset config key that can be set globally under `[general]` or per `[[datasets]]`, with per-dataset values overriding the global default.
- Apply `caption_prefix` to captions from sidecar text files and metadata JSONL rows before text encoder caching and training consume them.
- Preserve diffusion-pipe-style raw concatenation semantics: the configured string is prepended exactly as written, so users include any desired trailing space or comma.
- When a directory-backed dataset has `caption_prefix` but no matching sidecar for an item, use the prefix itself as that item's caption instead of dropping/failing the item.
- Avoid stale text encoder cache reuse when prefixed captions change and `--skip_existing` is enabled.
- Expose/document the option in dataset docs and keep the GUI TOML export path from silently dropping it.

## Capabilities

### New Capabilities
- `dataset-caption-prefix`: Covers global and per-dataset caption prefixing, missing-sidecar fallback behavior, and safe text-cache reuse for prefixed captions.

### Modified Capabilities
- None.

## Impact

- Shared dataset config schema and blueprint generation in `src/musubi_tuner/dataset/config_utils.py`.
- Shared datasource/dataset caption materialization in `src/musubi_tuner/dataset/image_video_dataset.py`.
- Text encoder cache skipping behavior in `src/musubi_tuner/cache_text_encoder_outputs.py` and architecture-specific callers.
- Dataset documentation in `docs/dataset_config.md`.
- GUI dataset schema/export/frontend fields under `src/musubi_tuner/gui_dashboard/`.
- Focused CPU tests for config fallback, sidecar/JSONL prefixing, missing-sidecar fallback, and stale text-cache detection.
