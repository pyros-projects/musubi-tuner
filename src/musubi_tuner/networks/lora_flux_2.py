# LoRA module for FLUX.2

import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


# Scan the full Flux2 transformer surface instead of only the block classes.
#
# This keeps Flux2 training/inference compatible with "all layer" LoRAs that also
# touch img/txt input projections, timestep embedding, modulation linears, and the
# final projection head. The generic LoRA helper will still only attach to Linear /
# Conv2d modules, so this remains a boring "all linear layers except excluded ones"
# policy.
FLUX_2_TARGET_REPLACE_MODULES = None


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
    # add default exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # LoRA on norm layers tends to be noisy and is not used by the extracted
    # Comfy / diffusion-pipe Flux2 LoRAs we want to stay compatible with.
    exclude_patterns.append(r".*(norm).*")

    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        FLUX_2_TARGET_REPLACE_MODULES,
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
        FLUX_2_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
