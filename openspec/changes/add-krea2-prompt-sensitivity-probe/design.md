## Context

Krea2 uses Qwen3-VL hidden states as stacked text conditioning: the text encoder returns selected Qwen layers shaped as `(batch, text_len, selected_layers, hidden)`, and the DiT's `TextFusionTransformer` mixes those layers through `txtfusion.layerwise_blocks`, `txtfusion.projector`, and `txtfusion.refiner_blocks`. The local Krea2 filter-bypass file is a direct patch for `txtfusion.projector`, so the useful diagnostic is not a generic text classifier; it is a measurement of how much Krea2's own fused text conditioning changes when that projector patch is applied.

The ltx-musubi repo already has the required primitives:

- `musubi_tuner.krea2.krea2_utils.load_krea2_text_encoder`
- `musubi_tuner.krea2.krea2_sampling.gather_valid_text`
- `musubi_tuner.krea2.krea2_mmdit.TextFusionTransformer`
- `musubi_tuner.modules.attention.AttentionParams`

The script belongs under `scripts/` with the existing sidecar utilities. It should be runnable from the repo root with `python scripts/krea2_prompt_sensitivity.py ...`, so it should make the local `src/` package importable when needed.

## Goals / Non-Goals

**Goals:**

- Provide a standalone operator script that scans individual prompts, prompt lists, and image sidecar folders.
- Produce an easy-to-read table plus CSV/JSON export for review.
- Score each prompt by comparing baseline Krea2 text-fusion output with patched text-fusion output.
- Reuse existing Krea2 text encoder and text-fusion code instead of duplicating model semantics.
- Keep the default path lighter than image generation by avoiding VAE load and full denoising.
- Add CPU-testable coverage for parsing, patch normalization, report formatting, and score math.

**Non-Goals:**

- Do not claim to detect safety policy or acceptance with certainty.
- Do not generate images or run the full Krea2 denoiser in the initial implementation.
- Do not change Krea2 training, cache formats, LoRA export, or dataset caption semantics.
- Do not add a new dependency for table formatting or report export.

## Decisions

### Use "prompt sensitivity" terminology

The report will describe prompts as stable, mild, sensitive, or bypass-sensitive. It will not call the score "safety" or "acceptance" because the measurement is only the conditioning delta caused by a known projector patch.

Alternative considered: name it an acceptance or safety score. Rejected because false certainty would be actively misleading; prompts can fail for reasons unrelated to the projector patch, and sensitive prompts can still render.

### Reuse Krea2 text-fusion rather than heuristic token matching

The scoring path will encode prompts through Qwen3-VL, compact valid tokens with `gather_valid_text`, run `TextFusionTransformer`, apply the bypass diff to a cloned or restored projector state, run it again, and compute deltas from the resulting fused contexts.

Alternative considered: maintain a deny/sensitive keyword list. Rejected because the observed behavior is phrase-context dependent and specifically tied to Krea2's text-fusion layer mixing.

### Load only text-fusion weights from the DiT checkpoint

The script should instantiate `TextFusionTransformer` using the existing Krea2 config and load only state dict keys under `txtfusion.` from the DiT safetensors. This avoids loading the large main DiT blocks, VAE, or sampler.

Alternative considered: load the full Krea2 DiT through `load_krea2_dit` and call the normal forward. Rejected for the default path because it would make a caption review tool consume training-scale VRAM. A future optional denoiser mode can revisit this if needed.

### Treat the bypass file as a direct projector patch

The script will support the known bypass format with key `diffusion_model.txtfusion.projector.diff`. It will normalize that path to `txtfusion.projector.weight` and add the diff at a configurable strength, defaulting to `1.0`.

Alternative considered: support arbitrary LoRA or checkpoint merge formats in the initial script. Rejected to keep this change focused; Krea2 training already has broader LoRA conversion paths.

### Keep input provenance in every report row

Collected prompts will carry source labels:

- `cli:<index>` for repeated `--prompt`
- `<file>:<line>` for `--prompt-file`
- `<path>` for sidecar `.txt` files

This lets operators jump back to the exact sidecar or prompt line after sorting by score.

Alternative considered: report prompt text only. Rejected because dataset cleanup is the main use case.

### Use standard library report formatting

Stdout will use a simple fixed-width table. Export will use standard-library CSV and JSON writers. The script should infer export format from `--output` extension, with an explicit `--format` override.

Alternative considered: add `rich` or `tabulate`. Rejected because this is an operator utility and the repo does not need another dependency for fixed-width output.

## Risks / Trade-offs

- Heuristic score thresholds can be wrong -> Expose the raw metrics and keep verdict labels coarse.
- Text-fusion-only deltas may not perfectly predict final images -> Document the score as bypass sensitivity and leave a future full-denoiser mode out of scope.
- Qwen3-VL still consumes meaningful memory -> Allow CPU encoding via `--device cpu`, support small `--batch-size`, and avoid loading the main DiT.
- Partial safetensors loading may miss checkpoint key variants -> Normalize both native and `diffusion_model.`-prefixed Krea2 keys and fail with a clear error if required `txtfusion` weights are missing.
- Sidecar folders may contain non-caption `.txt` files -> Default to all `.txt` files as requested, with deterministic sorting and optional recursive traversal.
