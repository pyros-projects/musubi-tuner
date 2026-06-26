"""CPU-testable Boogu Image tensor helpers."""

from __future__ import annotations

import math
import os
from typing import Any, Iterable

import torch


BOOGU_VAE_SCALE_FACTOR = 8
BOOGU_PATCH_SIZE = 2
BOOGU_VAE_SCALING_FACTOR = 0.3611
BOOGU_VAE_SHIFT_FACTOR = 0.1159
BOOGU_AUTOENCODER_KL_SINGLE_FILE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
    "up_block_types": ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
    "block_out_channels": (128, 256, 512, 512),
    "layers_per_block": 2,
    "latent_channels": 16,
    "sample_size": 1024,
    "scaling_factor": BOOGU_VAE_SCALING_FACTOR,
    "shift_factor": BOOGU_VAE_SHIFT_FACTOR,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
}


def load_boogu_autoencoder_kl(path: str, dtype: torch.dtype, autoencoder_kl_cls: Any | None = None):
    if autoencoder_kl_cls is None:
        from diffusers import AutoencoderKL

        autoencoder_kl_cls = AutoencoderKL

    if path.endswith(".safetensors"):
        return autoencoder_kl_cls.from_single_file(
            path,
            torch_dtype=dtype,
            **BOOGU_AUTOENCODER_KL_SINGLE_FILE_CONFIG,
        )
    if os.path.exists(os.path.join(path, "config.json")):
        return autoencoder_kl_cls.from_pretrained(path, torch_dtype=dtype)
    return autoencoder_kl_cls.from_pretrained(path, subfolder="vae", torch_dtype=dtype)


def pad_instruction_features(
    instruction_features: Iterable[torch.Tensor],
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = list(instruction_features)
    if not features:
        raise ValueError("instruction_features must contain at least one tensor")
    if any(feature.dim() != 2 for feature in features):
        raise ValueError("each instruction feature tensor must have shape (tokens, hidden)")

    hidden_sizes = {int(feature.shape[1]) for feature in features}
    if len(hidden_sizes) != 1:
        raise ValueError("all instruction feature tensors must have the same hidden size")

    max_tokens = max(int(feature.shape[0]) for feature in features)
    hidden_size = hidden_sizes.pop()
    target_device = torch.device(device) if device is not None else features[0].device
    target_dtype = dtype if dtype is not None else features[0].dtype

    padded = torch.zeros((len(features), max_tokens, hidden_size), device=target_device, dtype=target_dtype)
    mask = torch.zeros((len(features), max_tokens), device=target_device, dtype=torch.bool)
    for index, feature in enumerate(features):
        token_count = int(feature.shape[0])
        if token_count == 0:
            continue
        padded[index, :token_count] = feature.to(device=target_device, dtype=target_dtype)
        mask[index, :token_count] = True
    return padded, mask


def _lin_shift(num_tokens: float, x1: float = 256.0, y1: float = 0.5, x2: float = 4096.0, y2: float = 1.15) -> float:
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope * num_tokens + intercept


def boogu_time_schedule(num_steps: int, num_patch_tokens: int, device: torch.device | str | None = None) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    t_arr = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32, device=device)[:-1]
    mu = _lin_shift(max(1, int(num_patch_tokens)))
    eps = 1e-8
    t1 = torch.clamp(1.0 - t_arr, eps, 1.0 - eps)
    numerator = math.exp(mu)
    denominator = numerator + (1.0 / t1 - 1.0)
    shifted = 1.0 - numerator / denominator
    return torch.cat([shifted, torch.ones(1, dtype=torch.float32, device=shifted.device)])


def musubi_timestep_to_boogu_time(timestep: torch.Tensor) -> torch.Tensor:
    return 1.0 - timestep.to(dtype=torch.float32) / 1000.0


def boogu_raw_velocity_to_musubi_velocity(raw_velocity: torch.Tensor) -> torch.Tensor:
    return -raw_velocity


def boogu_loss_target(noise: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    return (noise - latents).detach()
