# LTX-2.5 support (readiness: code-complete, awaiting weights)

Status 2026-08-11: all code paths implemented and unit-tested (295-test suite
green), but **no real 2.5 checkpoint has been loaded yet** — the release files
were still downloading. First-load, embedding parity vs ComfyUI, and a smoke
training run are the remaining validation steps.

## What changed in 2.5 (verified against checkpoint headers)

- **DiT**: near-identical to 2.3 — same 48-block trunk, zero shape changes.
  Only deltas: video feed-forwards lose their biases (`ff_bias: false`), one
  new `keyframes_abs_pos_embedding` tensor `[1, 4096]`
  (`use_keyframes_abs_pos_embedding: true`), and `text_encoder_norm_type` is
  now spelled `PER_TOKEN_RMS` (no consumer in this repo — informational).
- **Packaging**: the all-in-one single file is gone; transformer, video VAE,
  audio VAE(+vocoder) and text encoder ship as separate files. The
  `video-vae-conv` variant is architecturally identical to 2.3; the default
  video VAE has a new diffusion-transformer decoder this repo does not
  implement (encoder side — the part used for latent caching — is unchanged
  in both variants).
- **Text encoder**: Gemma-4 unified 12B (`gemma4_unified`), unsupported by
  the pinned transformers. Same hidden size (3840) and the same 49-state ×
  per-token-RMS × dual-projection caption interface as 2.3.

## New/changed code

- `src/musubi_tuner/ltx2_repack.py` — merges the 2.5 split files back into a
  2.3-layout single checkpoint (streamed, ~64MB RSS; fail-closed slot
  validation; rejects quantized variants and the diffusion-decoder VAE).
- `ltx_2/model/transformer/*` — `ff_bias` and `use_keyframes_abs_pos_embedding`
  config support (2.3 defaults preserved; the keyframes parameter is
  load-compat only, keyframe conditioning itself is not implemented).
- `ltx_2/text_encoders/gemma/gemma4_text_model.py` — native Gemma-4 text
  tower (ported against ComfyUI's implementation): 5:1 sliding/full pattern,
  global layers with `attention_k_eq_v` (no v_proj), partial RoPE, per-layer
  scalars, unscaled QK-normed attention. Auto-detected from the checkpoint's
  `gemma_config` metadata; tokenizer loads from the embedded
  `tokenizer_json`. Prompt enhancement (`generate`) is not supported on this
  path.

## Once downloads finish

1. **Repack** (needs ~45GB free disk):

   ```bash
   .venv/bin/python src/musubi_tuner/ltx2_repack.py \
     --transformer /path/ltx-2.5-22b-dev-transformer-bf16.safetensors \
     --video_vae   /path/ltx-2.5-video-vae-conv-bf16.safetensors \
     --audio_vae   /path/ltx-2.5-audio-vae-bf16.safetensors \
     --text_encoder /path/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
     --output      /path/ltx-2.5-22b-dev.safetensors
   ```

2. **Point the usual flow at it** — two changes vs a 2.3 run:
   - `--ltx2_checkpoint /path/ltx-2.5-22b-dev.safetensors`
   - replace `--gemma_root ... --gemma_load_in_8bit` with
     `--gemma_safetensors /path/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`

3. **Re-cache everything** (latents AND text): VAE and text encoder weights
   are retrained even where architectures match. 2.3 caches are invalid.

## Known limitations / open items

- **Gemma-4 supports bf16 and the comfy int8-convrot release file** (detected
  via `.comfy_quant` markers; quantized linears run through comfy-kitchen
  kernels with a pure-PyTorch fallback). Prefer int8-convrot on 24GB cards:
  ~13GB resident vs the bf16 file's ~24GB shared-memory spill. bnb 8-bit/4-bit
  flags are not supported on the native path.
- The 2.5 **diffusion-decoder video VAE** is unsupported; use the `-conv`
  file (that is also why the repack tool rejects the other one).
- `keyframes_abs_pos_embedding` loads but keyframe conditioning is not
  implemented (new 2.5 inference feature).
- The distilled `lora-450` has not been checked against the 2.3 `lora-384`
  handling.
- First real-checkpoint load, ComfyUI embedding-parity spot check, and a
  training smoke run are pending weights.
