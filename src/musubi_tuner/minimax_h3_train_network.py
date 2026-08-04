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
    latents: torch.Tensor | None = None,
) -> torch.Tensor:
    if height % 32 or width % 32:
        raise ValueError("H3 preview width and height must be divisible by 32")
    shape = (1, 24, 1, height // 16, width // 16)
    if latents is None:
        latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    elif tuple(latents.shape) != shape:
        raise ValueError(f"H3 preview latents must have shape {shape}, got {tuple(latents.shape)}")
    else:
        latents = latents.to(device=device, dtype=torch.float32)

    context = context.to(device=device, dtype=dtype)
    token_tags = token_tags.to(device=device, dtype=torch.long)
    audio = torch.empty((1, 32, 2, 0), device=device, dtype=dtype)
    sigmas = build_h3_sigma_schedule(sample_steps, video_flow_shift, device)
    for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
        prediction, _ = transformer(
            latents.to(dtype),
            audio,
            sigma.view(1),
            [context],
            [token_tags],
        )
        latents = latents + (sigma - sigma_next) * prediction.float()
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
            if sample.get("frame_count", 1) != 1:
                raise ValueError("H3 image previews require frame_count=1")
            sample["frame_count"] = 1
            sample.setdefault("width", 512)
            sample.setdefault("height", 512)
            sample.setdefault("sample_steps", 12)
            if sample["width"] % 32 or sample["height"] % 32:
                raise ValueError("H3 preview width and height must be divisible by 32")
            if sample.get("negative_prompt") is not None:
                raise ValueError("MiniMax H3 preview sampling does not support negative prompts")

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
        if frame_count != 1 or do_classifier_free_guidance:
            raise ValueError("MiniMax H3 training previews support only single-frame generation without CFG")
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
        ).cpu()

        parked = False
        try:
            transformer.park_main_block_weights_for_decode()
            parked = True
            vae.to(accelerator.device)
            pixels = vae.decode(latents.to(accelerator.device))
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
        help="LoRA targets: attn (104), attn_mlp (208, portable default), or full (258, checkpoint-layout specific)",
    )
    parser.set_defaults(timestep_sampling="shift", discrete_flow_shift=12.0, mixed_precision="bf16")
    return parser


def main():
    parser = minimax_h3_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    MiniMaxH3NetworkTrainer().train(args)


if __name__ == "__main__":
    main()
