# MiniMax H3 — Research, Three-Way Comparison, and Review of the Image-Only LoRA Implementation

- **Author:** Claude (session on branch `feat/h3-image-lora-smoke`, 2026-08-04)
- **Scope:** (1) web research on MiniMax H3, (2) comparison of this repo's H3 implementation against `/home/pyro/repos/ai-toolkit` and `/home/pyro/repos/Fizgig`, (3) review of the image-only (`T=1`) training + preview implementation on this branch.
- **Method:** full read of the branch diff (~1,400 lines, 11 files + 2 new test files, all uncommitted vs `main`) and the H3 module sources; three parallel research agents (web, ai-toolkit, Fizgig); test suite executed; smoke-run artifacts inspected.
- **State at review time:** branch has **zero commits** — the entire implementation is uncommitted working-tree changes on top of `70c626a`.

---

## TL;DR

This repo is the most correct and most production-shaped H3 image-LoRA trainer of the three sibling implementations. Every numeric convention matches the official release, 29/29 tests pass, and the smoke rig works end-to-end (coherent, seam-free 1024² previews at step 400). Two findings matter:

1. **F1 (high value):** our `T=1` VAE decode is exactly ai-toolkit's *pre-fix* code; they measured ~16 dB → ~35 dB roundtrip PSNR by duplicating the lone latent to `T=2` before decoding.
2. **F2:** the default `--lora_target_preset full` trains LoRA on `adaln_proj.linear`, which ComfyUI **pruned** inference builds silently drop (shape mismatch) — Fizgig measured ~50% likeness loss from exactly this.

Plus a license landmine: the H3 Community License **excludes the EU** (among others) from its grant of rights.

---

## 1. What MiniMax H3 is (web research summary)

- **33B dense single-stream DiT** generating video + native synchronized stereo audio. Open weights (BF16, ~124 GB) released **2026-08-03**. No official GitHub repo, no paper. Partitions: **FL2VA** (t2va + first/last-frame→VA; the trainable one) and **Ref2VA** (omni-reference). The hosted 2K upscaler (`H3-Regenerate-2K`) and prompt preprocessor (`Context-IR`) were **not** open-sourced → local inference is effectively 768p.
- Ground truth from `FL2VA/transformer/config.json` (verified match with our `model.py` defaults, every field): hidden 5376, 50 layers, 56 heads × 128 (inner 7168 > hidden), FFN 14336 (SwiGLU), video latents 24ch, audio latents 32ch, patch (1,2,2), text_dim 5120, token refiner 2 layers, time_embed_dim 2688, `adaln_out_features` 96768 = 18×5376 (~13B of the 33B lives in AdaLN), rope_inv_freq_len 16, eps 1e-5.
- **Conventions that break naive trainers** (documented in a community trainer's FIXES.md whose v1 had *rising* loss): timestep input is `t = 1 − σ` (t=1 clean); the model predicts **data-ward velocity `x₀ − ε`**; video and audio ride **separate sigma-shift schedules (12.0 / 3.0)** mapped from the same base time (confirmed in official `model_index.json` → `sigma_shift_scales`). Checkpoints are **guidance-distilled — no CFG**.
- **VAEs:** VisualVAE `f16t4d24` (16× spatial, 4× temporal, 24ch, temporally causal encoder, **36-layer ViT decoder**); AudioVAE is DAC/BigVGAN lineage, 32 kHz → 40 latents/s, stereo channels processed independently, fp32-sensitive.
- **Text encoder:** Qwen3-VL-32B, conditioning on the **unnormalized hidden state after language layer 50**, raw prompt tokens, no chat template, no special tokens.
- **"INT8 ConvRot" is community quantization** (ComfyUI ecosystem) — MiniMax shipped BF16 only. No web source documents the algorithm; the answer is in ai-toolkit's vendored code: **arXiv:2512.03673 "ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers"** — regular-Hadamard block rotation (group 256) + per-output-row symmetric int8. Matches our `Int8ConvRotConfig` defaults.
- **Ecosystem:** upstream kohya musubi-tuner has **no H3 support** (this fork is ahead). ostris/ai-toolkit supports it; ComfyUI has day-0 support; diffusers support exists only as unreleased PR #14355 (the one our decoder credits).

### ⚠️ License

The **MiniMax H3 Community License excludes four jurisdictions from its grant of rights: the EU, UK, South Korea, and USA** (new vs. MiniMax M3 in June), plus a US$20M-revenue commercial gate. Relevant to anything trained on or derived from H3 that gets published from Germany. Not a technical finding — but it needs an owner.

Key sources: `huggingface.co/MiniMaxAI/MiniMax-H3` (+ raw `config.json`, `model_index.json`), `minimax.io/news/minimax-h3-open-source`, `github.com/IAmIronMan42/MiniMax-H3-FineTuning` (FIXES.md), `docs.comfy.org/tutorials/video/minimax/minimax-h3`, ConvRot quant model cards (ethanfel, Gluttony10, Abiray).

---

## 2. Three-way comparison

Both siblings landed their H3 support on **2026-08-03 — the same day the weights dropped**. All three are day-one implementations.

| Dimension | **This repo** | **ai-toolkit** (ostris, `extensions_built_in/diffusion_models/minimax_h3/`) | **Fizgig** (shootthesound, `src/fizgig/minimax/`) |
|---|---|---|---|
| Scope | Image-only `T=1` **and** joint video+audio LoRA; previews | t2v, i2v, image, joint audio, sampling — fullest matrix | Image-only LoRA, experimental; **no sampling, no audio** |
| Base quantization | ComfyUI INT8 ConvRot (int8 forward, bounded-BF16-chunk dequant backward) | Same INT8 ConvRot, straight-through fake-quant → trains against deployed int8 numerics | bitsandbytes **NF4**; explicitly *refuses* ConvRot files |
| Timestep training distribution | Presets: `image` = krea2-shift + `preserve_distribution_shape` + `max_timestep 875` (launcher default), `h3_video` = native shift-12 | **Uniform over the shift-12 inference grid** | Logit-normal + resolution-dependent shift (≈1.65 @ 0.25MP) |
| LoRA targets (default) | `full` = 258 (blocks + refiner incl. `adaln_proj`) | Every Linear in the transformer (incl. patch/cond/final projections) | attn+mlp = 208; **adaLN deliberately excluded** |
| LoRA key format | kohya `lora_unet_*` | `transformer.*` ↔ `diffusion_model.*` (Comfy-native) | kohya `lora_unet_*` — **key-compatible with ours** |
| Audio-less samples | Images: **zero audio tokens**; videos w/o track: silent latents | Silence rows always ride along (excluded from loss) | Audio modules constructed but never executed |
| In-training previews | Yes — prompt-embed **precache** (32B TE never coexists with 33B DiT), transformer parked to CPU for decode | Yes — full pipeline, TE stays resident (~16 GB nvfp4) | None ("evaluate checkpoints in ComfyUI") |
| Resume | `--autoresume` + `scheduler.bin` step recovery + batch skip | Generic harness resume | None |
| Batch >1 | Serial per-sample forward (packed seq is shape-dependent) | Padded batch + key masking (pad rows masked as keys) | Hard-locked to 1 |
| Tests / docs | **29 passing tests + `docs/minimax_h3.md`** | Zero tests, zero example configs (UI-only) | Zero committed tests |

### Convergent evolution (independent cross-validation of our choices)

- All three encode a still as a **`T=1` latent via `[:, :, -1:]`** after the causal encoder (identical code shape in all three).
- All three condition on **unnormalized Qwen3-VL layer-50** states, no chat template, `add_special_tokens=False`. (Truncation differs: ours 1024, Fizgig 512, ai-toolkit unbounded — harmless at caption lengths.)
- All three carry `FRAME_PER_TOKEN (1,4,4,4,4)`, `FRAME_RESCALE 5/3`, spatial rope scale 32, and **digit-identical latent mean/std constants**.
- Our `krea2_shift` image preset and Fizgig's empirical shift fix are **the same formula** (mu 0.5→1.15 over 256→6400 tokens), independently arrived at. Fizgig's receipts: *"H3's `sigma_shift_video=12.0` is the sampler's schedule — training image LoRAs with it put 57% of steps at sigma>0.9 and only ~4% below 0.3, which produced structurally poor likeness at every checkpoint."* ai-toolkit trains on exactly that shift-12 grid — for image LoRAs our/Fizgig's schedule should meaningfully outperform it. The `.pyro/h3/train.sh` presets (`image` vs `h3_video`) A/B this on our own data.

### Where this repo is ahead

- **Prompt-embed precache** (`minimax_h3_sample_prompts_cache.pt` + `--precache_sample_prompts` / `--cache_sample_prompts_only`) — unique among the three; the strictest VRAM discipline.
- **ComfyUI-venv INT8 TE caching** (`load_comfy_clip`, run the cache module inside ComfyUI's Python) — sidesteps reimplementing ConvRot dequant for the TE. Fizgig refuses those files outright, noting naive `weight*scale` dequant of rotated weights "would silently produce a garbage encoder" — which validates delegating to ComfyUI's own loader.
- **Only implementation with tests and docs.** Also the only one with autoresume.
- Bucketing at 32 px (16 VAE × 2 patch) avoids the odd-latent problem Fizgig has to crop away at loss time (`RESOLUTION_STEPS=16` → e.g. 31-px latent, crops up to 16 px of image edge).

---

## 3. Review of the image-only training implementation (this branch)

### Verified correct

| Item | Evidence |
|---|---|
| Flow-matching conventions: `t = 1−σ` (`minimax_h3/model.py:514`), target `clean − noise` (`minimax_h3_train_network.py:337`), Euler `x += (σ−σ′)·v` (`minimax_h3_train_network.py:92`) | Matches official `model_index.json` shifts, ai-toolkit's bridge (they flip timesteps and negate predictions to fit their harness), and community FIXES.md. These exact conventions broke other trainers' first attempts. |
| `T=1` encode is architecturally sound | Causal convs make a lone frame identical to a video's first-frame latent slot; single-tap optimization at `video_vae.py:47-49`; `17n+5 → 5n+2` math consistent across utils/dataset/tests; identical approach in both sibling repos. |
| Audio-sigma mapping | `time_shift_sigma(σ_v, 12→3)` in `process_batch` matches the model's internal mapping (`model.py:513`) and ai-toolkit's `remap_sigma` — same formula. |
| Empty-audio forward | Zero audio tokens flow through the real packing path (empty slices, adaLN rows, `unpack_audio` all safe); covered by tiny-model fwd/bwd test. Loss zeroed with guard against `audio_loss_weight > 0` (`minimax_h3_train_network.py:340-344`); normalization consistent (`(v+0)/(1+0)`). Cleanest of the three approaches; dodges the "zero-placeholder audio poisons the audio head" failure mode. |
| LoRA preset mechanics | Verified against `networks/lora.py` (`fullmatch`, include-rescues-exclude); counts 104/208/258 check out; smoke run's manual patterns ≡ `attn` preset. |
| Sampling integration | Returns [0,1] pixels per `save_images_grid` contract; base handles RNG save/restore and block-swap switching; H3 adds `network.eval()` hooks the base lacks; park/restore has a correct failure path (tested). |
| Autoresume | State-dir discovery, `scheduler.bin` `last_epoch` recovery, `skip_first_batches` epoch math — tested. Caveat: batch-skip alignment assumes unchanged dataset/bucketing between runs (standard). |
| Empirical | `PYTHONPATH=src <musubi venv>/bin/python -m pytest tests/test_minimax_h3.py tests/test_minimax_h3_sampling.py tests/test_trainer_base_autoresume.py tests/test_sai_model_spec.py` → **29 passed in 10.2 s**. Three real runs from 2026-08-04 morning (`lucy`, `woman`, `woman-5f-h3shift12`, up to step 400); lucy step-400 preview at 1024² is coherent with **zero tile seams** (4×4 tiled decode). |

### Findings, ranked

**F1 — `T=1` lone-latent decode is out-of-distribution for the ViT decoder (high value, evidence-backed).**
`decode_single_frame_latent` (`minimax_h3/video_vae.py:358-363`) decodes a lone latent through the ViT and keeps the last raw frame. This is exactly ai-toolkit's code **before** their commit `88ac27f`, whose message quantifies the problem: *"a lone temporal token is OOD for the chunk-trained ViT decoder (visibly patchy output, ~16 dB vs ~35 dB roundtrip)."* Root cause: video encoding **never produces `T=1`** — the minimum is `T=2` (a 5-frame clip, `n=0` in `5n+2`), so the decoder has never seen a lone temporal token. Their fix: `decode(cat([z, z], dim=2))` — a 5-frame-still-video in latent space — keeping the aligned frame (raw index 3 in our decoder's un-dropped 8-frame output; equivalently frame 0 after reference frame-dropping). ~2× decode cost, one-line-ish change in `_decode_clip`/`decode_single_frame_latent`. **Recommend verifying first** with a CPU VAE roundtrip PSNR on a training image (won't touch the training GPU), then adopting. Note our previews look fine at a glance — but previews are exactly where subtle decode degradation misleads about LoRA quality.

**F2 — Default `--lora_target_preset full` includes `adaln_proj.linear`; the ecosystem punishes that.**
Fizgig's two documented reasons for excluding adaLN (`Fizgig src/fizgig/minimax/trainer.py:32-39`): (a) ComfyUI **pruned** builds replace the AdaLN input width — `adaln_proj.linear` is `[96768, 8]` there, not `[96768, 2688]` (same curve-table mechanism as our `adaln_t_table` path) — so full-model-trained adaLN LoRA keys are **dropped with one shape error per block at load**; (b) in their real run those dropped keys cost **~50% likeness**. Our launcher already defaults to `attn_mlp`, but the repo default (`minimax_h3_train_network.py:358`) and the doc example both produce `full`. Recommend flipping the default to `attn_mlp` or adding a doc warning about pruned-build portability. Side nit: the help text "full (258, ai-toolkit-style)" is imprecise — ai-toolkit additionally wraps `video/audio_patch_proj`, `condition_proj`, `time_embedder`, and both final-layer heads.

**F3 — Personal launcher path in library error message.**
`minimax_h3_train_network.py:145`: the stale-prompt-cache error says *"Run `.pyro/h3/train.sh` once to rebuild it"* — `.pyro/` is untracked and personal. Should name the generic command (`minimax_h3_cache_text_encoder_outputs.py --precache_sample_prompts --sample_prompts ... --cache_sample_prompts_only`).

**F4 — Minor notes (fine as-is, listed for completeness).**
- Per-prompt-file `discrete_flow_shift` is silently ignored by previews (`do_inference` deletes it and uses `args.video_flow_shift`); the base logs it as if used.
- Each preview prompt runs a full transformer CPU↔GPU park/restore cycle — fine for 1–2 prompts, slow for many.
- Preview decode uses fp16 *weights*; the released recipe (per ai-toolkit's vendored notes) is fp16 *autocast over fp32 weights*. Minor numeric deviation.
- With the `image` timestep preset (`max_timestep 875`), previews integrate through σ-regions the LoRA never trained. Expected (the frozen base owns high-σ structure) — but worth remembering when reading early previews.
- Trainer-side `load_sample_prompt_cache` intentionally skips TE metadata validation (trainer doesn't know the TE path); staleness is enforced at precache time. By design, documented here so nobody "fixes" it.
- `_get_resume_position` maps steps→(epoch, batches) assuming stable dataloader length across runs — standard limitation, worth a docstring line at most.

### Suggested next steps

1. Verify F1 with a CPU VAE roundtrip PSNR; adopt the duplicate-to-`T=2` decode if confirmed.
2. Decide the default LoRA preset (F2) + doc warning about pruned-build adaLN dropping.
3. Replace the `.pyro` path in the error message (F3).
4. **Commit the branch** — ~1,400 clean, tested lines are sitting uncommitted.
5. (Non-code) Get an owner for the license question — EU exclusion.

---

## Addendum (2026-08-04, after Codie's fixes)

All four findings were addressed in `9bdb82d` (default preset → `attn_mlp` ✓, generic error message ✓, `--image_audio_mode none|silent` ✓ — nice on-the-fly design, no recache needed). One correction to the F1 implementation, settled empirically:

**Measured VAE roundtrip PSNR** (real checkpoint weights, fp32 CPU, 256², two lucy images) of every raw ViT-decoder frame vs the input still:

| Decode path | IMG_0025 | IMG_1092 |
|---|---|---|
| Lone `T=1`, best frame (pre-fix code) | 20.2 dB | 15.8 dB |
| `cat([z,z])` → raw[0] (`9bdb82d`) | 20.1 dB | 16.5 dB |
| `cat([z,z])` → raw[3] (Claude's review hypothesis) | 20.3 dB | 15.8 dB |
| **`cat([z,z])` → raw[7] (last frame)** | **28.4 dB** | **24.9 dB** |

So the duplication context was right but the first-frame slice bought nothing over the old path — and the aligned-frame theory (raw[3]) was wrong too. The last raw frame wins by 8–9 dB. Fixed by keeping the duplication and selecting `[:, :, -1:]` in both `decode_single_frame_latent` and the tiled per-tile slice; test expectation updated accordingly.

Control finding: encoding a *true* 5-frame still video (2 genuine latents) decodes **worse** than the duplicated-z trick (best frame 18.8 dB) — the `T=1` latent carries full single-frame detail while real video latents spread information temporally. So duplicate+last-frame is the right preview-decode recipe regardless of cache mode; `image_frame_count=5` remains interesting only for DiT-distribution reasons.

Experiment script: session scratchpad `frame_align_psnr.py` (encode still → decode `T=1` lone, `T=2` duplicated, and true-5-frame encode; per-frame PSNR).
