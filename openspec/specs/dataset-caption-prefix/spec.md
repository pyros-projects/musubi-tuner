# dataset-caption-prefix Specification

## Purpose
Allow Musubi dataset configs to prepend stable trigger words or fixed prompt prefixes to captions without rewriting sidecar or JSONL caption files.

## Requirements
### Requirement: Dataset configs accept caption prefixes
The system SHALL allow dataset configs to define `caption_prefix` as a string in `[general]` and in individual `[[datasets]]` entries.

#### Scenario: General caption prefix applies to a dataset
- **WHEN** `[general]` defines `caption_prefix = "trigger "` and a dataset does not define `caption_prefix`
- **THEN** the dataset SHALL use `"trigger "` as its caption prefix

#### Scenario: Dataset caption prefix overrides general caption prefix
- **WHEN** `[general]` defines `caption_prefix = "global "` and a dataset defines `caption_prefix = "local "`
- **THEN** the dataset SHALL use `"local "` as its caption prefix

#### Scenario: Omitted caption prefix preserves current behavior
- **WHEN** neither `[general]` nor the dataset defines `caption_prefix`
- **THEN** the dataset SHALL behave as if `caption_prefix = ""`

### Requirement: Caption prefixes are applied to sidecar captions
The system SHALL prepend `caption_prefix` to captions loaded from directory sidecar text files before creating `ItemInfo.caption`.

#### Scenario: Sidecar caption is prefixed
- **WHEN** an item sidecar contains `"a person standing"` and the dataset has `caption_prefix = "middlesplits "`
- **THEN** the effective caption SHALL be `"middlesplits a person standing"`

#### Scenario: Prefix concatenation is exact
- **WHEN** an item sidecar contains `"a person standing"` and the dataset has `caption_prefix = "middlesplits,"`
- **THEN** the effective caption SHALL be `"middlesplits,a person standing"`

#### Scenario: Empty sidecar caption uses prefix
- **WHEN** an item sidecar exists but contains an empty caption and the dataset has `caption_prefix = "middlesplits"`
- **THEN** the effective caption SHALL be `"middlesplits"`

### Requirement: Caption prefixes are applied to JSONL captions
The system SHALL prepend `caption_prefix` to captions loaded from image, video, and audio JSONL metadata rows before creating `ItemInfo.caption`.

#### Scenario: JSONL caption is prefixed
- **WHEN** a JSONL row contains `"caption": "a person standing"` and the dataset has `caption_prefix = "middlesplits "`
- **THEN** the effective caption SHALL be `"middlesplits a person standing"`

#### Scenario: JSONL empty caption uses prefix
- **WHEN** a JSONL row contains `"caption": ""` and the dataset has `caption_prefix = "middlesplits"`
- **THEN** the effective caption SHALL be `"middlesplits"`

### Requirement: Missing sidecars can fall back to the caption prefix
The system SHALL allow directory-backed datasets with a non-empty `caption_prefix` to use the prefix itself as the caption when an item's sidecar text file is missing.

#### Scenario: Image without sidecar remains in dataset when prefix exists
- **WHEN** an image dataset has `caption_extension = ".txt"` and `caption_prefix = "middlesplits"` but an image has no matching `.txt` file
- **THEN** the image SHALL remain eligible for the dataset
- **AND** its effective caption SHALL be `"middlesplits"`

#### Scenario: Video without sidecar uses prefix
- **WHEN** a video dataset has `caption_extension = ".txt"` and `caption_prefix = "scene "` but a video has no matching `.txt` file
- **THEN** the video's effective caption SHALL be `"scene "`

#### Scenario: Audio without sidecar uses prefix
- **WHEN** an audio dataset has `caption_extension = ".txt"` and `caption_prefix = "audio "` but an audio file has no matching `.txt` file
- **THEN** the audio item's effective caption SHALL be `"audio "`

#### Scenario: Missing sidecar without prefix preserves existing strictness
- **WHEN** a directory-backed dataset has an empty `caption_prefix` and an item has no required sidecar
- **THEN** the dataset SHALL preserve the current missing-caption behavior for that media type

### Requirement: Text encoder cache skipping respects effective captions
The system SHALL NOT skip an existing text encoder cache file when its stored caption metadata differs from the current effective caption.

#### Scenario: Matching cached caption is skipped
- **WHEN** `--skip_existing` is enabled and an existing text encoder cache file has `caption1` metadata equal to the current effective caption
- **THEN** the cache script SHALL skip re-encoding that item

#### Scenario: Mismatched cached caption is refreshed
- **WHEN** `--skip_existing` is enabled and an existing text encoder cache file has `caption1` metadata different from the current effective caption
- **THEN** the cache script SHALL re-encode that item
- **AND** the saved cache metadata SHALL record the current effective caption

#### Scenario: Missing cached caption metadata is refreshed
- **WHEN** `--skip_existing` is enabled and an existing text encoder cache file has no `caption1` metadata
- **THEN** the cache script SHALL re-encode that item

### Requirement: Caption prefix configuration is documented and exported
The system SHALL expose `caption_prefix` in dataset documentation and GUI dataset TOML export paths.

#### Scenario: Documentation describes raw prefix semantics
- **WHEN** a user reads the dataset config documentation
- **THEN** it SHALL explain that `caption_prefix` is prepended exactly as configured and does not insert whitespace automatically

#### Scenario: GUI export preserves per-dataset prefix
- **WHEN** a GUI dataset entry has `caption_prefix = "middlesplits "`
- **THEN** exported dataset TOML SHALL include `caption_prefix = "middlesplits "`
