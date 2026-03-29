from __future__ import annotations

from typing import Iterable, Optional

import torch

from musubi_tuner import convert_lora

import logging

logger = logging.getLogger(__name__)


def normalize_lora_weight_args(
    lora_weights: Optional[Iterable[str]],
    lora_multipliers: Optional[Iterable[float] | float],
) -> tuple[list[str], list[float]]:
    weights = list(lora_weights or [])
    if not weights:
        return [], []

    if lora_multipliers is None:
        multipliers: list[float] = []
    elif isinstance(lora_multipliers, (int, float)):
        multipliers = [float(lora_multipliers)]
    else:
        multipliers = [float(multiplier) for multiplier in lora_multipliers]

    while len(multipliers) < len(weights):
        multipliers.append(1.0)
    if len(multipliers) > len(weights):
        multipliers = multipliers[: len(weights)]

    return weights, multipliers


def convert_lora_weight_keys_for_inference(
    weights_sd: dict[str, torch.Tensor],
    network_module_name: str,
) -> dict[str, torch.Tensor]:
    if not weights_sd:
        return weights_sd

    first_key = next(iter(weights_sd))
    if first_key.startswith("lora_"):
        return weights_sd
    if first_key.startswith("diffusion_model.") or first_key.startswith("transformer."):
        logger.info(f"Converting LoRA weights for {network_module_name} from diffusers format to default format.")
        return convert_lora.convert_from_diffusers("lora_unet_", weights_sd)
    return weights_sd


def normalize_flux_inference_lora_weights(
    weights_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return convert_lora_weight_keys_for_inference(weights_sd, "networks.lora_flux_2")


def normalize_zimage_inference_lora_weights(
    weights_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    weights_sd = convert_lora_weight_keys_for_inference(weights_sd, "networks.lora_zimage")

    packed_qkv_prefixes = sorted(
        {
            key[: -len(".lora_down.weight")]
            for key in weights_sd
            if key.endswith(".lora_down.weight") and "attention_qkv" in key
        }
    )
    if packed_qkv_prefixes:
        logger.info("Inference LoRA uses packed attention_qkv weights; splitting to to_q/to_k/to_v.")
        converted: dict[str, torch.Tensor] = {}
        skip_prefixes = set(packed_qkv_prefixes)
        for key, value in weights_sd.items():
            prefix = key.split(".", 1)[0]
            if prefix in skip_prefixes:
                continue
            converted[key] = value

        for prefix in packed_qkv_prefixes:
            down_key = f"{prefix}.lora_down.weight"
            up_key = f"{prefix}.lora_up.weight"
            alpha_key = f"{prefix}.alpha"
            down_weight = weights_sd[down_key]
            up_weight = weights_sd[up_key]
            alpha = weights_sd.get(alpha_key)
            up_chunks = torch.tensor_split(up_weight, 3, dim=0)
            for suffix, up_chunk in zip(("to_q", "to_k", "to_v"), up_chunks):
                new_prefix = prefix.replace("attention_qkv", f"attention_{suffix}")
                converted[f"{new_prefix}.lora_down.weight"] = down_weight.clone()
                converted[f"{new_prefix}.lora_up.weight"] = up_chunk.contiguous()
                if alpha is not None:
                    converted[f"{new_prefix}.alpha"] = alpha.clone() if torch.is_tensor(alpha) else alpha
        weights_sd = converted

    for key, value in list(weights_sd.items()):
        if not key.endswith(".lora_down.weight"):
            continue
        prefix = key[: -len(".lora_down.weight")]
        alpha_key = f"{prefix}.alpha"
        if alpha_key not in weights_sd:
            weights_sd[alpha_key] = torch.tensor(float(value.shape[0]), dtype=torch.float32)

    return weights_sd
