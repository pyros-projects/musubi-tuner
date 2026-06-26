## ADDED Requirements

### Requirement: Prompt sensitivity script is available
The system SHALL provide a standalone Krea2 prompt-sensitivity script under `scripts/` that can be run from the repository root.

#### Scenario: Script runs from repository root
- **WHEN** an operator runs `python scripts/krea2_prompt_sensitivity.py --help` from the repository root
- **THEN** the command SHALL display usage for the Krea2 prompt-sensitivity probe

#### Scenario: Script uses local package modules
- **WHEN** the script is run directly from the repository root
- **THEN** it SHALL be able to import the local `musubi_tuner` package from `src/`

### Requirement: Prompt inputs are collected from supported sources
The system SHALL collect prompts from repeated CLI prompt arguments, one-prompt-per-line prompt files, and folders of `.txt` sidecar files.

#### Scenario: CLI prompts are collected
- **WHEN** the script is invoked with multiple `--prompt` arguments
- **THEN** each prompt SHALL appear as a separate report row
- **AND** each row SHALL include a `cli:<index>` source label

#### Scenario: Prompt file lines are collected
- **WHEN** the script is invoked with `--prompt-file prompts.txt`
- **THEN** each non-empty line in `prompts.txt` SHALL appear as a separate report row
- **AND** each row SHALL include a `<path>:<line>` source label

#### Scenario: Sidecar folder prompts are collected
- **WHEN** the script is invoked with `--sidecar-dir <folder>`
- **THEN** each `.txt` file directly under `<folder>` SHALL be read as a prompt source
- **AND** each row SHALL include the sidecar file path as its source label

#### Scenario: Recursive sidecar collection is optional
- **WHEN** the script is invoked with `--sidecar-dir <folder> --recursive`
- **THEN** `.txt` files in nested folders SHALL also be collected

#### Scenario: Blank prompts are ignored
- **WHEN** a CLI prompt, file line, or sidecar file resolves to an empty prompt after trimming
- **THEN** it SHALL NOT produce a report row

### Requirement: Scores compare baseline and patched Krea2 text fusion
The system SHALL score each prompt by comparing Krea2 text-fusion conditioning with and without an applied bypass projector patch.

#### Scenario: Baseline and patched conditioning are compared
- **WHEN** the script scores a prompt
- **THEN** it SHALL encode the prompt with the Krea2 Qwen3-VL conditioner
- **AND** it SHALL compact valid text tokens with the existing Krea2 valid-text gathering behavior
- **AND** it SHALL run the Krea2 text-fusion module before and after applying the bypass patch
- **AND** it SHALL compute numeric delta metrics from the two fused conditioning tensors

#### Scenario: Bypass patch is applied to the projector
- **WHEN** the bypass safetensors contains `diffusion_model.txtfusion.projector.diff`
- **THEN** the script SHALL add that tensor to the Krea2 `txtfusion.projector.weight` at the configured strength

#### Scenario: Missing bypass patch fails clearly
- **WHEN** the bypass safetensors does not contain a supported projector diff key
- **THEN** the script SHALL fail with a clear error explaining that the bypass patch could not be found

#### Scenario: Missing text-fusion weights fail clearly
- **WHEN** the Krea2 DiT checkpoint does not provide the required `txtfusion` weights
- **THEN** the script SHALL fail with a clear error explaining that text-fusion weights could not be loaded

### Requirement: Report includes metrics, verdicts, and provenance
The system SHALL produce a tabular report that includes enough information to review and edit prompts.

#### Scenario: Stdout report is printed
- **WHEN** the script finishes scoring prompts
- **THEN** it SHALL print a table to stdout containing source, prompt, token count, numeric sensitivity metrics, aggregate score, and verdict

#### Scenario: Report can be sorted by score
- **WHEN** the script is invoked with score sorting enabled
- **THEN** the highest-scoring prompts SHALL appear first in the stdout table and exported report

#### Scenario: Verdict labels are coarse sensitivity labels
- **WHEN** a prompt is scored
- **THEN** the script SHALL assign a coarse sensitivity verdict based on configured thresholds
- **AND** the verdict SHALL NOT be described as a definitive safety or acceptance classification

#### Scenario: Main score uses absolute cosine redirection
- **WHEN** the script scores prompts
- **THEN** the aggregate score SHALL be based on absolute cosine redirection caused by the bypass patch
- **AND** the score SHALL NOT require operator-provided or built-in baseline prompt controls
- **AND** prompts whose patched conditioning direction changes more SHALL receive higher scores

### Requirement: Report can be exported
The system SHALL export the prompt-sensitivity report to CSV or JSON when requested.

#### Scenario: CSV export is written
- **WHEN** the script is invoked with `--output report.csv`
- **THEN** it SHALL write a CSV file containing the same report rows and metrics printed to stdout

#### Scenario: JSON export is written
- **WHEN** the script is invoked with `--output report.json`
- **THEN** it SHALL write a JSON file containing the same report rows and metrics printed to stdout

#### Scenario: Export format can be explicit
- **WHEN** the script is invoked with `--output report.txt --format csv`
- **THEN** it SHALL write CSV output despite the non-CSV extension

### Requirement: Probe avoids full Krea2 image generation by default
The system SHALL keep the initial prompt-sensitivity probe limited to text encoding and text fusion, without loading the VAE or running full denoising.

#### Scenario: Default probe does not load the VAE
- **WHEN** the script runs in its default mode
- **THEN** it SHALL NOT require a VAE checkpoint path

#### Scenario: Default probe does not generate images
- **WHEN** the script runs in its default mode
- **THEN** it SHALL NOT run the Krea2 denoising sampler
- **AND** it SHALL NOT write image outputs
