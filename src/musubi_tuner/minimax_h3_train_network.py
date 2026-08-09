from __future__ import annotations

import argparse
import logging
import os
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_MINIMAX_H3, ARCHITECTURE_MINIMAX_H3_FULL
from musubi_tuner.hv_train_network import NetworkTrainer, read_config_from_file, setup_parser_common
from musubi_tuner.minimax_h3 import minimax_h3_utils, sampling_lora_overlay
from musubi_tuner.minimax_h3.convert_lora_to_comfy import convert_lora_to_comfy
from musubi_tuner.utils import huggingface_utils
from musubi_tuner.minimax_h3.model import MiniMaxH3Model, time_shift_sigma
from musubi_tuner.minimax_h3.video_vae import load_video_vae_decoder
from musubi_tuner.minimax_h3_cache_text_encoder_outputs import (
    load_sample_prompt_cache,
    resolve_sample_prompts_cache_path,
)
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

H3_LORA_TARGET_PRESETS: dict[str, list[str] | None] = {
    "attn": [r".*\.attn\.(qkv_proj|out_proj)$"],
    "attn_mlp": [r".*\.attn\.(qkv_proj|out_proj)$", r".*\.mlp\.(fc1|fc2)$"],
    "no_packed_attn": [
        r"token_refiner\.blocks\.[0-9]+\.attn\.(qkv_proj|out_proj)$",
        r"token_refiner\.blocks\.[0-9]+\.mlp\.(fc1|fc2)$",
        r"blocks\.[0-9]+\.mlp\.(fc1|fc2)$",
    ],
    # Main-stream MLPs only — no token_refiner. For style/rendering concepts whose
    # tag-soup captions otherwise drift the text path (fox->cup bleed at refiner
    # heat ~2x median, mandalas 2026-08-09).
    "mlp": [r"blocks\.[0-9]+\.mlp\.(fc1|fc2)$"],
    "full": None,
}


def _apply_h3_lora_target_preset(args: argparse.Namespace) -> None:
    preset = args.lora_target_preset
    network_args = list(args.network_args or [])
    if any(arg.startswith(("exclude_patterns=", "include_patterns=")) for arg in network_args):
        logger.warning("Explicit LoRA include/exclude patterns override lora_target_preset=%s", preset)
        return

    include_patterns = H3_LORA_TARGET_PRESETS[preset]
    if include_patterns is not None:
        # This branch's generic LoRA include patterns override excludes rather than forming an allow-list.
        network_args.extend(["exclude_patterns=['.*']", f"include_patterns={include_patterns!r}"])
    args.network_args = network_args or None


def build_h3_sigma_schedule(sample_steps: int, shift: float, device: torch.device) -> torch.Tensor:
    if sample_steps < 1:
        raise ValueError("H3 sample_steps must be at least 1")
    if shift <= 0:
        raise ValueError("H3 video flow shift must be positive")
    base = torch.linspace(1.0, 0.0, sample_steps + 1, device=device, dtype=torch.float32)
    return time_shift_sigma(base, 1.0, shift)


def resolve_preview_solver(solver: str, sample_steps: int) -> str:
    """AB2's multistep extrapolation overshoots on few-step schedules (step ratios
    approach 2.0 under shift-12), frying the latents — observed with the 4-step
    Turbo overlay. Fall back to euler below 8 steps."""
    if solver == "ab2" and sample_steps < 8:
        logger.warning("sample_solver=ab2 is unstable at %d steps; using euler for this preview", sample_steps)
        return "euler"
    return solver


def _parse_overlay_spec(spec: str) -> tuple[str, float]:
    """Split an overlay spec ``PATH[:STRENGTH]`` (Windows drive colons survive)."""
    path, _, tail = spec.rpartition(":")
    try:
        strength = float(tail) if path else 1.0
    except ValueError:
        path, strength = spec, 1.0
    if not path:
        path = spec
    return path, strength


def _silent_audio_latent_length(video_latent_frames: int) -> int:
    if video_latent_frames == 1:
        source_frames = 1
    elif video_latent_frames >= 2 and (video_latent_frames - 2) % 5 == 0:
        source_frames = (video_latent_frames - 2) // 5 * minimax_h3_utils.VIDEO_FRAME_STRIDE + minimax_h3_utils.VIDEO_FRAME_OFFSET
    else:
        raise ValueError(f"Invalid H3 video latent length for silent audio: {video_latent_frames}")
    return minimax_h3_utils.audio_latent_length(source_frames)


@torch.no_grad()
def sample_h3_image_latents(
    transformer,
    context: torch.Tensor,
    token_tags: torch.Tensor,
    *,
    height: int,
    width: int,
    sample_steps: int,
    video_flow_shift: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    audio_flow_shift: float = 3.0,
    latent_frames: int = 2,
    audio_latent_frames: int = 0,
    solver: str = "ab2",
    latents: torch.Tensor | None = None,
) -> torch.Tensor:
    if height % 32 or width % 32:
        raise ValueError("H3 preview width and height must be divisible by 32")
    if latent_frames != 1 and (latent_frames < 2 or (latent_frames - 2) % 5):
        raise ValueError("H3 previews support 1 temporal latent or the video contract 5n+2")
    if solver not in ("euler", "ab2"):
        raise ValueError(f"Unknown H3 preview solver: {solver}")
    shape = (1, 24, latent_frames, height // 16, width // 16)
    if latents is None:
        latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    elif tuple(latents.shape) != shape:
        raise ValueError(f"H3 preview latents must have shape {shape}, got {tuple(latents.shape)}")
    else:
        latents = latents.to(device=device, dtype=torch.float32)

    context = context.to(device=device, dtype=dtype)
    token_tags = token_tags.to(device=device, dtype=torch.long)
    if audio_latent_frames > 0:
        audio = torch.randn((1, 32, 2, audio_latent_frames), generator=generator, device=device, dtype=torch.float32)
    else:
        audio = torch.zeros((1, 32, 2, 0), device=device, dtype=torch.float32)
    sigmas = build_h3_sigma_schedule(sample_steps, video_flow_shift, device)
    audio_sigmas = time_shift_sigma(sigmas, video_flow_shift, audio_flow_shift)

    previous_video = previous_audio = None
    previous_step = previous_audio_step = None
    step_bar = tqdm(
        range(sample_steps),
        desc=f"H3 preview {width}x{height} lat={latent_frames} {solver}",
        leave=False,
        dynamic_ncols=True,
    )
    for index in step_bar:
        video_step = (sigmas[index] - sigmas[index + 1]).item()
        audio_step = (audio_sigmas[index] - audio_sigmas[index + 1]).item()
        sampling_lora_overlay.update_time_state(transformer, sigmas[index])
        pred_video, pred_audio = transformer(
            latents.to(dtype),
            audio.to(dtype),
            sigmas[index].view(1),
            [context],
            [token_tags],
        )
        pred_video = pred_video.float()
        pred_audio = pred_audio.float()
        if solver == "ab2" and previous_video is not None:
            # Variable-step Adams-Bashforth 2 on the velocity field.
            ratio = video_step / previous_step
            latents = latents + video_step * ((1.0 + ratio / 2.0) * pred_video - (ratio / 2.0) * previous_video)
            if audio.numel():
                audio_ratio = audio_step / previous_audio_step
                audio = audio + audio_step * ((1.0 + audio_ratio / 2.0) * pred_audio - (audio_ratio / 2.0) * previous_audio)
        else:
            latents = latents + video_step * pred_video
            if audio.numel():
                audio = audio + audio_step * pred_audio
        previous_video, previous_audio = pred_video, pred_audio
        previous_step, previous_audio_step = video_step, audio_step
    return latents


class MiniMaxH3NetworkTrainer(NetworkTrainer):
    @property
    def architecture(self) -> str:
        return ARCHITECTURE_MINIMAX_H3

    def on_post_save(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        ckpt_name: str,
        save_dtype,
        metadata: dict,
        force_sync_upload: bool,
    ) -> None:
        if getattr(args, "no_convert_to_comfy", False) or not ckpt_name.endswith(".safetensors"):
            return
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        comfy_name = ckpt_name[: -len(".safetensors")] + ".comfy.safetensors"
        comfy_file = os.path.join(args.output_dir, comfy_name)
        try:
            convert_lora_to_comfy(ckpt_file, comfy_file)
        except Exception as e:
            accelerator.print(f"Warning: ComfyUI LoRA conversion failed for {ckpt_file}: {e}")
            return
        accelerator.print(f"saved ComfyUI-format LoRA: {comfy_file}")
        if args.huggingface_repo_id is not None:
            huggingface_utils.upload(args, comfy_file, "/" + comfy_name, force_sync_upload=force_sync_upload)

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_MINIMAX_H3_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        if args.fp8_base or args.fp8_scaled:
            raise ValueError("MiniMax H3 does not support the FP8 flags. Pass a Comfy INT8 ConvRot checkpoint to --dit directly.")
        if args.mixed_precision not in (None, "bf16"):
            raise ValueError("MiniMax H3 training requires --mixed_precision bf16")
        args.mixed_precision = "bf16"
        if args.video_flow_shift <= 0 or args.audio_flow_shift <= 0:
            raise ValueError("MiniMax H3 flow shifts must be positive")
        if args.audio_loss_weight < 0:
            raise ValueError("--audio_loss_weight must be non-negative")
        sample_blocks_to_swap = getattr(args, "sample_blocks_to_swap", None)
        if sample_blocks_to_swap is not None:
            if sample_blocks_to_swap < 0:
                raise ValueError("--sample_blocks_to_swap must be non-negative for MiniMax H3")
            if sample_blocks_to_swap > 48:
                raise ValueError("--sample_blocks_to_swap cannot exceed 48 for MiniMax H3")
            if sample_blocks_to_swap > 0 and not getattr(args, "blocks_to_swap", 0):
                raise ValueError(
                    "Positive --sample_blocks_to_swap requires positive --blocks_to_swap to initialize MiniMax H3's "
                    "classic block-swap offloader"
                )
        cfg_scale = float(getattr(args, "cfg_augmented_scale", 1.0) or 1.0)
        if cfg_scale < 1.0:
            raise ValueError(f"--cfg_augmented_scale must be >= 1.0 (1.0 = off): {cfg_scale}")
        if cfg_scale > 1.0 and getattr(args, "train_lora_overlay", None):
            raise ValueError("--cfg_augmented_scale and --train_lora_overlay both counter distillation loss; use one")
        train_overlay = getattr(args, "train_lora_overlay", None)
        if train_overlay:
            overlay_path, overlay_strength = _parse_overlay_spec(train_overlay)
            if not os.path.isfile(overlay_path):
                raise ValueError(f"--train_lora_overlay file not found: {overlay_path}")
            if overlay_strength <= 0:
                raise ValueError(f"--train_lora_overlay strength must be positive: {train_overlay}")
        if getattr(args, "sample_prompts", None) and getattr(args, "block_swap_h2d_only", False):
            raise ValueError("H3 preview sampling currently requires classic block swap; omit --block_swap_h2d_only")
        _apply_h3_lora_target_preset(args)
        self.dit_dtype = torch.bfloat16
        args.dit_dtype = "bfloat16"
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0
        self.default_discrete_flow_shift = args.video_flow_shift
        self.vae_frame_stride = 17

    def process_sample_prompts(self, args, accelerator, sample_prompts):
        samples = load_prompts(sample_prompts)
        for sample in samples:
            frame_count = int(sample.get("frame_count", 1))
            if frame_count != 1:
                aligned = minimax_h3_utils.align_frame_count(frame_count, "down")
                if aligned != frame_count:
                    logger.warning("H3 preview frame_count %d aligned down to %d (17*n+5)", frame_count, aligned)
                frame_count = aligned
            sample["frame_count"] = frame_count
            sample.setdefault("width", 512)
            sample.setdefault("height", 512)
            sample.setdefault("sample_steps", 12)
            if sample["width"] % 32 or sample["height"] % 32:
                raise ValueError("H3 preview width and height must be divisible by 32")
            if sample.get("negative_prompt") is not None:
                raise ValueError("MiniMax H3 preview sampling does not support negative prompts")
            if int(sample.get("sample_latent_frames", 2)) not in (1, 2):
                raise ValueError("Prompt-level sample_latent_frames must be 1 or 2")
            if sample.get("sample_audio_mode", "auto") not in ("auto", "none", "silent"):
                raise ValueError("Prompt-level sample_audio_mode must be auto, none, or silent")
            if sample.get("sample_solver", "ab2") not in ("euler", "ab2"):
                raise ValueError("Prompt-level sample_solver must be euler or ab2")
            if sample.get("sample_frame_select", "dup_last") not in ("dup_last", "first", "last", "sharpest"):
                raise ValueError("Prompt-level sample_frame_select must be dup_last, first, last, or sharpest")

        cache_path = resolve_sample_prompts_cache_path(args.dataset_config)
        cached_prompts = load_sample_prompt_cache(cache_path, samples)
        if cached_prompts is None:
            raise ValueError(
                f"H3 sample prompt cache is missing or stale: {cache_path}. "
                "Rebuild it with minimax_h3_cache_text_encoder_outputs.py using "
                "--precache_sample_prompts --sample_prompts and --cache_sample_prompts_only before training starts."
            )
        for sample, cached in zip(samples, cached_prompts):
            sample["h3_text_embed"] = cached["h3_text_embed"]
            sample["h3_token_tags"] = cached["h3_token_tags"]
        logger.info("Loaded %d H3 sample prompt embedding(s) from %s", len(samples), cache_path)
        return samples

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        del guidance_scale, cfg_scale, image_path, control_video_path
        if do_classifier_free_guidance:
            raise ValueError("MiniMax H3 training previews do not support CFG")
        # The base trainer rounds frame_count with a stride contract H3 does not
        # follow, so trust the aligned value from process_sample_prompts.
        del frame_count
        frame_count = int(sample_parameter.get("frame_count", 1))
        if frame_count == 1:
            # Per-prompt sampling overrides beat the CLI flags so one run can A/B configurations.
            latent_frames = int(sample_parameter.get("sample_latent_frames", getattr(args, "sample_latent_frames", 2)))
        else:
            latent_frames = minimax_h3_utils.video_latent_length(frame_count)
        audio_mode = sample_parameter.get("sample_audio_mode", getattr(args, "sample_audio_mode", "auto"))
        if audio_mode == "auto":
            audio_mode = getattr(args, "image_audio_mode", "none")
        audio_latent_frames = _silent_audio_latent_length(latent_frames) if audio_mode == "silent" else 0
        overlays = getattr(self, "_sampling_overlays", None) or []
        # Per-prompt overlay control: floats scale the configured strength (0 = off);
        # bools/strings keep the legacy on/off semantics.
        raw_overlay = sample_parameter.get("sample_lora_overlay", 1)
        try:
            overlay_scale = float(raw_overlay)
        except (TypeError, ValueError):
            overlay_scale = 1.0 if str(raw_overlay).lower() in ("true", "yes", "on") else 0.0
        overlay_enabled = bool(overlays) and overlay_scale > 0.0
        solver = resolve_preview_solver(sample_parameter.get("sample_solver", getattr(args, "sample_solver", "ab2")), sample_steps)
        sample_started = time.perf_counter()
        try:
            if overlay_enabled:
                stats = sampling_lora_overlay.attach_sampling_lora_overlays(
                    transformer,
                    [(modules, strength * overlay_scale) for modules, strength in overlays],
                    temb_grid=getattr(self, "_sampling_overlay_grid", None),
                )
                if not getattr(self, "_sampling_overlay_logged", False):
                    self._sampling_overlay_logged = True
                    accelerator.print(
                        f"Sampling LoRA overlay active: {stats['backbone']} backbone, "
                        f"{stats['adaln_grid']} adaln-grid, {len(stats['skipped'])} skipped modules"
                    )
            latents = sample_h3_image_latents(
                transformer,
                sample_parameter["h3_text_embed"],
                sample_parameter["h3_token_tags"],
                height=height,
                width=width,
                sample_steps=sample_steps,
                video_flow_shift=discrete_flow_shift,
                device=accelerator.device,
                dtype=dit_dtype,
                generator=generator,
                audio_flow_shift=args.audio_flow_shift,
                latent_frames=latent_frames,
                audio_latent_frames=audio_latent_frames,
                solver=solver,
            ).cpu()
        finally:
            if overlay_enabled:
                sampling_lora_overlay.clear_sampling_lora_overlays(transformer)
        sample_done = time.perf_counter()

        parked = False
        try:
            transformer.park_main_block_weights_for_decode()
            parked = True
            vae.to(accelerator.device)
            if frame_count == 1:
                frame_select = sample_parameter.get("sample_frame_select", getattr(args, "sample_frame_select", "dup_last"))
                pixels = vae.decode(latents.to(accelerator.device), frame_select=frame_select)
            else:
                # Sampled audio latents are discarded; video previews save as silent mp4.
                pixels = vae.decode_video(latents.to(accelerator.device))
            pixels = ((pixels.float() + 1.0) * 0.5).clamp_(0.0, 1.0).cpu()
            sample_seconds = sample_done - sample_started
            decode_seconds = time.perf_counter() - sample_done
            overlay_desc = f"{overlays[0][1] * overlay_scale:.2f}" if overlay_enabled else "off"
            logger.info(
                "H3 preview done: %dx%d frames=%d latents=%d steps=%d solver=%s overlay=%s | sample %.1fs + decode %.1fs = %.1fs",
                width, height, frame_count, latent_frames, sample_steps, solver,
                overlay_desc, sample_seconds, decode_seconds, sample_seconds + decode_seconds,
            )
            return pixels
        finally:
            vae.to("cpu")
            if accelerator.device.type == "cuda":
                torch.cuda.empty_cache()
            if parked:
                transformer.restore_main_block_weights_after_decode()

    def load_vae(self, args, vae_dtype, vae_path):
        if not vae_path:
            raise ValueError("--vae is required for MiniMax H3 preview sampling")
        return load_video_vae_decoder(
            vae_path,
            device="cpu",
            dtype=vae_dtype,
            disable_numpy_memmap=args.disable_numpy_memmap,
        )

    def _load_sampling_overlays(self, args, accelerator) -> None:
        if hasattr(self, "_sampling_overlays"):
            return
        self._sampling_overlays = []
        self._sampling_overlay_grid = None
        for spec in getattr(args, "sample_lora_overlay", None) or []:
            path, strength = _parse_overlay_spec(spec)
            modules = sampling_lora_overlay.normalize_overlay_state_dict(sampling_lora_overlay.load_overlay_file(path))
            self._sampling_overlays.append((modules, strength))
            accelerator.print(f"Loaded sampling LoRA overlay: {path} ({len(modules)} modules, strength {strength})")
        grid_path = getattr(args, "sample_lora_overlay_temb_grid", None)
        if self._sampling_overlays and grid_path:
            self._sampling_overlay_grid = sampling_lora_overlay.load_temb_grid(grid_path)

    def _ensure_train_overlay(self, args, accelerator, transformer) -> None:
        """Attach the frozen training-time assistant LoRA (e.g. the ostris
        de-distillation adapter) so training forwards run on the assisted field.

        Called from process_batch, so the attach always lands after the trainable
        network has wrapped module forwards — clearing restores the trained-LoRA
        wrapper, never the bare module. Previews detach it (inference must show
        the un-assisted base the LoRA will actually be used on) and the next
        training step lazily re-attaches.
        """
        spec = getattr(args, "train_lora_overlay", None)
        if not spec or getattr(self, "_train_overlay_attached", False):
            return
        if not hasattr(self, "_train_overlay"):
            path, strength = _parse_overlay_spec(spec)
            modules = sampling_lora_overlay.normalize_overlay_state_dict(sampling_lora_overlay.load_overlay_file(path))
            self._train_overlay = (modules, strength)
            accelerator.print(f"Loaded training LoRA overlay: {path} ({len(modules)} modules, strength {strength})")
        stats = sampling_lora_overlay.attach_sampling_lora_overlays(accelerator.unwrap_model(transformer), [self._train_overlay])
        self._train_overlay_attached = True
        if not getattr(self, "_train_overlay_logged", False):
            self._train_overlay_logged = True
            accelerator.print(
                f"Training LoRA overlay active: {stats['backbone']} backbone, {len(stats['skipped'])} skipped modules "
                "(detached for previews, never merged into saves)"
            )

    def _detach_train_overlay(self, accelerator, transformer) -> None:
        if getattr(self, "_train_overlay_attached", False):
            sampling_lora_overlay.clear_sampling_lora_overlays(accelerator.unwrap_model(transformer))
            self._train_overlay_attached = False

    @torch.no_grad()
    def _adapter_cosine(self, network, transformer) -> float | None:
        """Global cosine between the trained LoRA delta and the attached training
        adapter's delta (Gram cross-terms; deltas never materialized). Negative =
        the LoRA is absorbing an anti-adapter component that corrupts the plain
        base at inference — the de-distillation collapse fingerprint."""
        if not getattr(self, "_train_overlay_attached", False):
            return None
        loras_by_name = {lora.lora_name: lora for lora in getattr(network, "unet_loras", None) or []}
        if not loras_by_name:
            return None
        num = l2 = a2 = None
        for path, module in transformer.named_modules():
            entries = sampling_lora_overlay.get_module_adapters(module)
            lora = loras_by_name.get("lora_unet_" + path.replace(".", "_")) if entries else None
            if lora is None or getattr(lora, "lora_up", None) is None:
                continue
            ul, dl = lora.lora_up.weight.float(), lora.lora_down.weight.float()
            if ul.ndim != 2 or dl.ndim != 2:
                continue
            for entry in entries:
                ua, da = entry["up"], entry["down"]
                if ua.device != ul.device or entry.get("offset") is not None:
                    continue
                uaf, daf = ua.float(), da.float()
                inner = ((ul.T @ uaf) * (dl @ daf.T)).sum()
                ln = ((ul.T @ ul) * (dl @ dl.T)).sum()
                an = ((uaf.T @ uaf) * (daf @ daf.T)).sum()
                num = inner if num is None else num + inner
                l2 = ln if l2 is None else l2 + ln
                a2 = an if a2 is None else a2 + an
        if num is None:
            return None
        denominator = (l2 * a2).clamp_min(1e-24).sqrt()
        return float((num / denominator).item())

    def extra_postfix_logs(self, args, accelerator, network, transformer) -> dict:
        cosine = self._adapter_cosine(accelerator.unwrap_model(network), accelerator.unwrap_model(transformer))
        self._last_adapter_cos = cosine
        logs = {} if cosine is None else {"acos": round(cosine, 4)}
        drift = getattr(self, "_last_guidance_drift", None)
        if drift is not None:
            logs["gd"] = round(drift, 3)
        return logs

    def extra_step_logs(self, args, logs) -> dict:
        extra = {}
        cosine = getattr(self, "_last_adapter_cos", None)
        if cosine is not None:
            extra["network/adapter_cos"] = cosine
        drift = getattr(self, "_last_guidance_drift", None)
        if drift is not None:
            extra["network/guidance_drift"] = drift
        return extra

    def on_before_sample_images(self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype):
        self._detach_train_overlay(accelerator, transformer)
        self._sample_network_was_training = network.training
        network.eval()
        self._load_sampling_overlays(args, accelerator)

    def on_after_sample_images(self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype):
        # Defensive: per-prompt attach/clear in do_inference already restores forwards.
        # The training overlay is NOT re-attached here — the next process_batch does it lazily.
        if getattr(self, "_sampling_overlays", None):
            sampling_lora_overlay.clear_sampling_lora_overlays(accelerator.unwrap_model(transformer))
        network.train(getattr(self, "_sample_network_was_training", True))

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: torch.dtype | None,
    ):
        if dit_weight_dtype not in (None, torch.bfloat16):
            raise ValueError(f"MiniMax H3 compute must use BF16, got {dit_weight_dtype}")
        model = minimax_h3_utils.load_transformer(
            dit_path,
            device=loading_device,
            dtype=torch.bfloat16,
            attn_mode=attn_mode,
            split_attn=split_attn,
            disable_numpy_memmap=args.disable_numpy_memmap,
        )
        model.sigma_shift_video = args.video_flow_shift
        model.sigma_shift_audio = args.audio_flow_shift
        if getattr(args, "offload_checkpoint_activations", False):
            model.enable_activation_offload()
            accelerator.print("Checkpoint activation offload to pinned CPU enabled (Unsloth-style)")
        return model

    def compile_transformer(self, args, transformer):
        model: MiniMaxH3Model = transformer
        return model_utils.compile_transformer(
            args, model, [model.token_refiner.blocks, model.blocks], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents):
        # Both H3 VAEs store already-normalized latents in the cache.
        return latents

    def call_dit(
        self,
        args,
        accelerator,
        transformer,
        latents,
        batch,
        noise,
        noisy_model_input,
        timesteps,
        network_dtype,
        **kwargs,
    ):
        raise RuntimeError("MiniMax H3 uses its joint audio/video process_batch implementation")

    @staticmethod
    def _apply_cfg_augmentation(pred: torch.Tensor, uncond: torch.Tensor, scale: float) -> torch.Tensor:
        """Recover the raw conditional velocity from the guidance-distilled output
        (diffusion-pipe technique): fit ``(pred + (scale-1)*uncond) / scale`` — the
        rearranged CFG equation — against the ordinary flow target, uncond carrying
        no gradient. At convergence the raw output stays ``uncond + scale*(target -
        uncond)``, i.e. the distilled convention is preserved. (The amplifying
        direction is exactly wrong: it trains the model to strip its guidance.)
        """
        return (pred + (scale - 1.0) * uncond) / scale

    def _get_uncond_context(self, args, accelerator, network_dtype):
        cached = getattr(self, "_cfg_uncond_context", None)
        if cached is None:
            from musubi_tuner.minimax_h3_cache_text_encoder_outputs import (
                DEFAULT_UNCOND_CACHE,
                load_sample_prompt_cache,
                resolve_sample_prompts_cache_path,
            )

            cache_path = resolve_sample_prompts_cache_path(args.dataset_config, filename=DEFAULT_UNCOND_CACHE)
            entries = load_sample_prompt_cache(cache_path, [{"prompt": ""}])
            if entries is None:
                raise ValueError(
                    f"--cfg_augmented_scale needs the uncond embedding cache at {cache_path}; "
                    "re-run the H3 text-encoder cache step with --precache_uncond"
                )
            embed = entries[0]["h3_text_embed"].to(accelerator.device, dtype=network_dtype)
            tags = entries[0]["h3_token_tags"].to(accelerator.device, dtype=torch.long)
            cached = self._cfg_uncond_context = (embed, tags)
            accelerator.print(f"Loaded H3 uncond embedding for CFG-augmented training ({embed.shape[0]} tokens)")
        return cached

    @torch.no_grad()
    def _measure_guidance_drift(self, args, accelerator, transformer, network, noisy_video, noisy_audio, sigma_video, contexts, tags):
        """Guidance-retention gauge: ``||cond - uncond||`` of the LoRA'd model over
        the frozen base, on the current batch. 1.0 = distillation intact; standard
        (non-CFG-augmented) training drifts it toward 1/s as the model un-distills.
        """
        uncond_embed, uncond_tags = self._get_uncond_context(args, accelerator, network_dtype=noisy_video.dtype)
        uncond_contexts = [uncond_embed] * len(contexts)
        uncond_tag_list = [uncond_tags] * len(contexts)
        model = accelerator.unwrap_model(transformer)
        raw_network = accelerator.unwrap_model(network)
        offloader = getattr(model, "offloader", None) if getattr(model, "blocks_to_swap", 0) else None
        if offloader is not None:
            offloader.set_forward_only(True)
        try:
            with accelerator.autocast():
                cond_lora, _ = transformer(noisy_video, noisy_audio, sigma_video, contexts, tags)
                uncond_lora, _ = transformer(noisy_video, noisy_audio, sigma_video, uncond_contexts, uncond_tag_list)
                raw_network.set_multiplier(0.0)
                try:
                    cond_base, _ = transformer(noisy_video, noisy_audio, sigma_video, contexts, tags)
                    uncond_base, _ = transformer(noisy_video, noisy_audio, sigma_video, uncond_contexts, uncond_tag_list)
                finally:
                    raw_network.set_multiplier(1.0)
        finally:
            if offloader is not None:
                offloader.set_forward_only(False)
        separation_lora = (cond_lora.float() - uncond_lora.float()).norm()
        separation_base = (cond_base.float() - uncond_base.float()).norm().clamp_min(1e-8)
        return float((separation_lora / separation_base).item())

    @staticmethod
    def _weighted_mse(pred, target, sigma, weighting_scheme):
        loss = F.mse_loss(pred, target, reduction="none")
        if weighting_scheme == "sigma_sqrt":
            weight = sigma.float().clamp_min(1e-6).pow(-2)
        elif weighting_scheme == "cosmap":
            weight = 2 / (torch.pi * (1 - 2 * sigma.float() + 2 * sigma.float().square()))
        else:
            weight = None
        if weight is not None:
            loss = loss * weight.view(-1, *([1] * (loss.ndim - 1)))
        return loss.mean()

    def process_batch(
        self,
        args,
        accelerator,
        transformer,
        network,
        batch,
        latents,
        noise,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        vae,
        global_step,
    ):
        if "latents_audio" not in batch or "h3_text_embed" not in batch:
            raise ValueError("H3 audio and text caches are missing; run both minimax_h3 cache commands first")
        self._ensure_train_overlay(args, accelerator, transformer)
        sigma_video, _timesteps = self.get_noisy_model_input_and_timesteps(
            args,
            noise,
            latents,
            batch["timesteps"],
            noise_scheduler,
            accelerator.device,
            dit_dtype,
            return_sigmas=True,
        )

        clean_video = latents.to(accelerator.device, dtype=network_dtype)
        noise_video = noise.to(accelerator.device, dtype=network_dtype)
        clean_audio = batch["latents_audio"].to(accelerator.device, dtype=network_dtype)
        has_audio_target = clean_audio.numel() > 0
        if not has_audio_target and args.image_audio_mode == "silent":
            audio_frames = _silent_audio_latent_length(clean_video.shape[2])
            clean_audio = clean_audio.new_zeros((*clean_audio.shape[:-1], audio_frames))
        noise_audio = torch.randn_like(clean_audio)
        sigma_video = sigma_video.to(accelerator.device)
        sigma_audio = time_shift_sigma(sigma_video, args.video_flow_shift, args.audio_flow_shift)
        sv = sigma_video.view(-1, 1, 1, 1, 1).to(network_dtype)
        sa = sigma_audio.view(-1, 1, 1, 1).to(network_dtype)
        noisy_video = (1 - sv) * clean_video + sv * noise_video
        noisy_audio = (1 - sa) * clean_audio + sa * noise_audio

        contexts = [value.to(accelerator.device, dtype=network_dtype) for value in batch["h3_text_embed"]]
        tags = [value.to(accelerator.device, dtype=torch.long) for value in batch.get("h3_token_tags", [])]
        if not tags:
            tags = [torch.ones(value.shape[0], device=accelerator.device, dtype=torch.long) for value in contexts]
        if args.gradient_checkpointing:
            noisy_video.requires_grad_(True)
            noisy_audio.requires_grad_(True)
            for value in contexts:
                value.requires_grad_(True)

        drift_every = int(getattr(args, "guidance_drift_every", 0) or 0)
        if drift_every > 0 and global_step > 0 and global_step % drift_every == 0 and not getattr(self, "_drift_measured_at", None) == global_step:
            self._drift_measured_at = global_step
            self._last_guidance_drift = self._measure_guidance_drift(
                args, accelerator, transformer, network, noisy_video, noisy_audio, sigma_video, contexts, tags
            )

        cfg_scale = float(getattr(args, "cfg_augmented_scale", 1.0) or 1.0)
        if cfg_scale > 1.0:
            # Fused walk: cond (grad, checkpointed) and empty-prompt (no-grad)
            # streams share one pass through the swapped blocks — same math as
            # two forwards at a single walk's swap cost, no residency toggles.
            uncond_embed, uncond_tags = self._get_uncond_context(args, accelerator, network_dtype)
            with accelerator.autocast():
                pred_video, pred_audio, uncond_video, uncond_audio = accelerator.unwrap_model(transformer).forward_with_uncond(
                    noisy_video, noisy_audio, sigma_video, contexts, tags, uncond_embed, uncond_tags
                )
            pred_video = self._apply_cfg_augmentation(pred_video, uncond_video, cfg_scale)
            pred_audio = self._apply_cfg_augmentation(pred_audio, uncond_audio, cfg_scale)
        else:
            with accelerator.autocast():
                pred_video, pred_audio = transformer(noisy_video, noisy_audio, sigma_video, contexts, tags)
        target_video = clean_video - noise_video
        target_audio = clean_audio - noise_audio
        video_loss = self._weighted_mse(pred_video.to(network_dtype), target_video, sigma_video, args.weighting_scheme)
        if not has_audio_target:
            if args.audio_loss_weight > 0:
                raise ValueError("Image-only H3 caches require --audio_loss_weight 0")
            audio_loss = video_loss.new_zeros(())
            loss = video_loss
        else:
            audio_loss = self._weighted_mse(pred_audio.to(network_dtype), target_audio, sigma_audio, args.weighting_scheme)
            loss = (video_loss + args.audio_loss_weight * audio_loss) / (1.0 + args.audio_loss_weight)
        return loss, {"loss/video": float(video_loss.detach()), "loss/audio": float(audio_loss.detach())}


def minimax_h3_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--video_flow_shift", type=float, default=12.0, help="H3 video sigma shift")
    parser.add_argument("--audio_flow_shift", type=float, default=3.0, help="H3 audio sigma shift")
    parser.add_argument("--audio_loss_weight", type=float, default=1.0, help="relative audio reconstruction loss weight")
    parser.add_argument(
        "--image_audio_mode",
        choices=("none", "silent"),
        default="none",
        help="empty image audio rows (none) or duration-matched noised zero-audio rows (silent)",
    )
    parser.add_argument(
        "--lora_target_preset",
        choices=list(H3_LORA_TARGET_PRESETS),
        default="attn_mlp",
        help=(
            "LoRA targets: attn (104), attn_mlp (208, portable default), "
            "no_packed_attn (108), mlp (100, main-stream MLPs only — no text refiner), "
            "or full (258, checkpoint-layout specific)"
        ),
    )
    parser.add_argument(
        "--sample_latent_frames",
        type=int,
        choices=(1, 2),
        default=2,
        help="temporal latents for H3 previews: 2 samples the minimum natural video packet, 1 is the legacy single-latent mode",
    )
    parser.add_argument(
        "--sample_audio_mode",
        choices=("auto", "none", "silent"),
        default="auto",
        help="preview audio rows: silent adds duration-matched audio tokens, auto follows --image_audio_mode",
    )
    parser.add_argument(
        "--sample_solver",
        choices=("euler", "ab2"),
        default="ab2",
        help="preview ODE solver; ab2 is a second-order multistep and sharper at low step counts",
    )
    parser.add_argument(
        "--sample_frame_select",
        choices=("dup_last", "first", "last", "sharpest"),
        default="dup_last",
        help="preview frame from the two-latent packet: dup_last decodes the second latent through the measured-best duplicate path; first/last/sharpest pick from the natural five-frame clip",
    )
    parser.add_argument(
        "--no_convert_to_comfy",
        action="store_true",
        help="skip writing the ComfyUI-format .comfy.safetensors twin next to each saved LoRA",
    )
    parser.add_argument(
        "--sample_lora_overlay",
        action="append",
        metavar="PATH[:STRENGTH]",
        help="frozen LoRA applied only during preview sampling as a runtime overlay (repeatable); "
        "e.g. the MiniMax-H3 Turbo LoRA for 4-step previews. Accepts musubi, ComfyUI, and bare turbo key formats",
    )
    parser.add_argument(
        "--sample_lora_overlay_temb_grid",
        default=None,
        help="silu(t_emb) grid safetensors (from the ComfyUI-MiniMax-H3-Turbo node) enabling adaln overlay "
        "modules on pruned curve-table bases; without it adaln modules are skipped with a warning",
    )
    parser.add_argument(
        "--cfg_augmented_scale",
        type=float,
        default=1.0,
        help="CFG-augmented training (diffusion-pipe): fit (pred + (s-1)*uncond)/s — the de-amplified raw velocity — "
        "with a no-grad empty-prompt forward so training preserves the model's guidance distillation. 1.0 = off; "
        "4.0 recommended. Needs the --precache_uncond cache; mutually exclusive with --train_lora_overlay",
    )
    parser.add_argument(
        "--offload_checkpoint_activations",
        action="store_true",
        help="offload gradient-checkpoint boundary activations to pinned CPU RAM (Unsloth-style): frees "
        "~blocks x [B,L,hidden] of VRAM for batch-size headroom at a little PCIe traffic",
    )
    parser.add_argument(
        "--guidance_drift_every",
        type=int,
        default=0,
        help="every N steps, measure ||cond-uncond|| of the LoRA'd model vs the frozen base (4 no-grad forwards) and "
        "log gd= / network/guidance_drift. 1.0 = distillation intact, drifts toward 1/s as training un-distills. "
        "Needs the --precache_uncond cache. 0 = off",
    )
    parser.add_argument(
        "--train_lora_overlay",
        type=str,
        default=None,
        help="frozen assistant LoRA overlaid during TRAINING forwards, PATH[:STRENGTH] — e.g. the ostris "
        "minimax_h3_training_adapter (de-distillation). Detached for previews and never merged into saves, "
        "so saved LoRAs apply to the plain base at inference",
    )
    parser.set_defaults(timestep_sampling="shift", discrete_flow_shift=12.0, mixed_precision="bf16")
    return parser


def main():
    parser = minimax_h3_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    MiniMaxH3NetworkTrainer().train(args)


if __name__ == "__main__":
    main()
