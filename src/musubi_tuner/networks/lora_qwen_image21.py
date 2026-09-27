# LoRA module for Qwen-Image 2.1

from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


# None (rather than a list of module class names) makes the generic LoRA walker wrap every
# Linear in the DiT: per block to_q/to_k/to_v/to_out.0 and the fused SwiGLU gate_up/out, plus
# img_in, txt_in, the timestep MLP, the shared modulation, norm_out.linear and proj_out.
# Narrow it with include_patterns / exclude_patterns in --network_args, e.g. blocks only:
#   include_patterns=['transformer_blocks\..*']
#
# The DiT's module names equal the Comfy checkpoint keys, so the saved lora_unet_* keys load in
# ComfyUI as they are (including the fused img_mlp.gate_up of INT8 ConvRot files).
QWEN_IMAGE21_TARGET_REPLACE_MODULES = None


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    return lora.create_network(
        QWEN_IMAGE21_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        QWEN_IMAGE21_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
