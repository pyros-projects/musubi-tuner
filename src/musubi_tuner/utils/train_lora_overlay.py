"""Frozen, unmerged LoRA overlays on the DiT for training (``--train_lora_overlay PATH[:STRENGTH]``)."""

import re
from types import ModuleType

import torch


def parse_overlay_spec(spec: str) -> tuple[str, float]:
    """``PATH`` or ``PATH:STRENGTH`` (H3's --train_lora_overlay convention)."""
    match = re.fullmatch(r"(.+):([0-9]*\.?[0-9]+)", spec)
    return (match.group(1), float(match.group(2))) if match else (spec, 1.0)


def attach_train_lora_overlay(
    model: torch.nn.Module,
    network_module: ModuleType,
    weights: dict[str, torch.Tensor],
    strength: float,
    device,
    dtype: torch.dtype,
    source: str,
):
    """Hook a frozen LoRA (musubi ``lora_unet_*`` keys) onto the DiT without merging it.

    Every adapted Linear computes W x + strength * B A x. A trained LoRA applied afterwards wraps
    these modules and learns on top. The overlay is not part of the trained network, so it never
    reaches the optimizer or a save.
    """
    network = network_module.create_arch_network_from_weights(strength, weights, unet=model, for_inference=True)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    info = network.load_state_dict(weights, strict=False)
    if info.missing_keys or info.unexpected_keys:
        raise ValueError(f"LoRA overlay {source} does not match the DiT: {info}")
    network.to(device=device, dtype=dtype)
    network.requires_grad_(False)
    return network


def set_train_lora_overlay_enabled(network, enabled: bool) -> None:
    for lora in network.unet_loras:
        lora.enabled = enabled
