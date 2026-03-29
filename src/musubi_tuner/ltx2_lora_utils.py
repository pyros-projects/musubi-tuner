from __future__ import annotations

from typing import Dict

import torch


def looks_like_comfy_ltx_lora(weights_sd: Dict[str, torch.Tensor]) -> bool:
    if not weights_sd:
        return False
    keys = list(weights_sd.keys())
    return (
        all(key.startswith("diffusion_model.") for key in keys)
        and any(".lora_A.weight" in key for key in keys)
        and any(".lora_B.weight" in key for key in keys)
    )


def normalize_ltx_comfy_lora_weights(weights_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted: Dict[str, torch.Tensor] = {}
    synthesized_alpha: Dict[str, torch.Tensor] = {}

    for key, value in weights_sd.items():
        if not key.startswith("diffusion_model."):
            converted[key] = value
            continue

        if key.endswith(".lora_A.weight"):
            module_path = key[len("diffusion_model.") : -len(".lora_A.weight")]
            target_key = f"lora_unet_model_{module_path.replace('.', '_')}.lora_down.weight"
            converted[target_key] = value
            alpha_key = f"lora_unet_model_{module_path.replace('.', '_')}.alpha"
            synthesized_alpha.setdefault(alpha_key, torch.tensor(float(value.shape[0]), dtype=torch.float32))
            continue

        if key.endswith(".lora_B.weight"):
            module_path = key[len("diffusion_model.") : -len(".lora_B.weight")]
            target_key = f"lora_unet_model_{module_path.replace('.', '_')}.lora_up.weight"
            converted[target_key] = value
            continue

        if key.endswith(".alpha"):
            module_path = key[len("diffusion_model.") : -len(".alpha")]
            target_key = f"lora_unet_model_{module_path.replace('.', '_')}.alpha"
            converted[target_key] = value
            continue

        converted[key] = value

    for key, value in synthesized_alpha.items():
        converted.setdefault(key, value)

    return converted
