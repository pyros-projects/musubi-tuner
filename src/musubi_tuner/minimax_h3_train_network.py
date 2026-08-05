from __future__ import annotations

import argparse
import logging

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_MINIMAX_H3, ARCHITECTURE_MINIMAX_H3_FULL
from musubi_tuner.hv_train_network import NetworkTrainer, read_config_from_file, setup_parser_common
from musubi_tuner.minimax_h3 import minimax_h3_utils
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
    for index in range(sample_steps):
        video_step = (sigmas[index] - sigmas[index + 1]).item()
        audio_step = (audio_sigmas[index] - audio_sigmas[index + 1]).item()
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
            solver=sample_parameter.get("sample_solver", getattr(args, "sample_solver", "ab2")),
        ).cpu()

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
            return ((pixels.float() + 1.0) * 0.5).clamp_(0.0, 1.0).cpu()
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

    def on_before_sample_images(self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype):
        self._sample_network_was_training = network.training
        network.eval()

    def on_after_sample_images(self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype):
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
            "no_packed_attn (108), or full (258, checkpoint-layout specific)"
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
    parser.set_defaults(timestep_sampling="shift", discrete_flow_shift=12.0, mixed_precision="bf16")
    return parser


def main():
    parser = minimax_h3_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    MiniMaxH3NetworkTrainer().train(args)


if __name__ == "__main__":
    main()
