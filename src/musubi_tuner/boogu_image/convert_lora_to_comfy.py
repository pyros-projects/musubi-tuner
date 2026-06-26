"""Convert Boogu Image LoRA checkpoints from Musubi format to native ComfyUI format."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


_BOOGU_PATH_TOKENS = (
    "double_stream_blocks",
    "single_stream_blocks",
    "context_refiner",
    "noise_refiner",
    "ref_img_refiner",
    "img_instruct_attn",
    "instruct_attn",
    "img_self_attn",
    "instruct_self_attn",
    "img_feed_forward",
    "instruct_feed_forward",
    "feed_forward",
    "time_caption_embed",
    "caption_projection",
    "norm_out",
    "proj_out",
    "to_out",
    "to_q",
    "to_k",
    "to_v",
    "linear_1",
    "linear_2",
    "linear_3",
)


def _consume_known_token(parts: list[str], start: int) -> tuple[str, int]:
    for token in sorted(_BOOGU_PATH_TOKENS, key=lambda value: value.count("_"), reverse=True):
        token_parts = token.split("_")
        end = start + len(token_parts)
        if parts[start:end] == token_parts:
            return token, end
    return parts[start], start + 1


def convert_lora_name_to_comfy_module(lora_name: str) -> str:
    prefix = "lora_unet_"
    if not lora_name.startswith(prefix):
        raise ValueError(f"Unsupported Boogu LoRA name: {lora_name}")

    parts = lora_name[len(prefix) :].split("_")
    converted: list[str] = []
    index = 0
    while index < len(parts):
        if parts[index].isdigit():
            converted.append(parts[index])
            index += 1
            continue
        token, index = _consume_known_token(parts, index)
        converted.append(token)
    return ".".join(converted)


def convert_key_to_comfy(key: str) -> Optional[str]:
    if key.endswith(".alpha"):
        return None

    try:
        lora_name, weight_name = key.split(".", 1)
    except ValueError as exc:
        raise ValueError(f"Unexpected Boogu LoRA key format: {key}") from exc

    if weight_name not in {"lora_down.weight", "lora_up.weight"}:
        raise ValueError(f"Unexpected Boogu LoRA weight key: {key}")

    module_path = convert_lora_name_to_comfy_module(lora_name)
    return f"diffusion_model.{module_path}.{weight_name}"


def convert_state_dict_to_comfy(weights_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    lora_alpha: dict[str, torch.Tensor] = {}
    lora_rank: dict[str, int] = {}

    for key, tensor in weights_sd.items():
        if key.endswith(".lora_down.weight"):
            lora_name = key[: -len(".lora_down.weight")]
            lora_rank[lora_name] = int(tensor.shape[0])
        elif key.endswith(".alpha"):
            lora_name = key[: -len(".alpha")]
            lora_alpha[lora_name] = tensor

    converted: dict[str, torch.Tensor] = {}
    for key, tensor in weights_sd.items():
        new_key = convert_key_to_comfy(key)
        if new_key is None:
            continue

        if key.endswith(".lora_up.weight"):
            lora_name = key[: -len(".lora_up.weight")]
            alpha = lora_alpha.get(lora_name)
            rank = lora_rank.get(lora_name)
            if alpha is not None and rank:
                scale = float(alpha.item()) / float(rank)
                if scale != 1.0:
                    tensor = tensor * scale

        converted[new_key] = tensor

    return converted


def convert_lora_to_comfy(input_path: str | os.PathLike, output_path: str | os.PathLike | None = None, verbose: bool = False) -> Path:
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.comfy{input_path.suffix}"
    else:
        output_path = Path(output_path)

    if verbose:
        logger.info("Loading Boogu LoRA from %s", input_path)
    weights_sd = load_file(str(input_path))
    converted = convert_state_dict_to_comfy(weights_sd)

    metadata = None
    with safe_open(str(input_path), framework="pt") as f:
        metadata = f.metadata()

    if verbose:
        logger.info("Saving Boogu ComfyUI LoRA to %s (%d keys)", output_path, len(converted))
    save_file(converted, str(output_path), metadata=metadata)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Boogu Image LoRA from Musubi format to native ComfyUI format")
    parser.add_argument("input", type=str, help="Path to the input Boogu LoRA checkpoint")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to save the converted LoRA")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print conversion details")
    args = parser.parse_args()

    convert_lora_to_comfy(args.input, args.output, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
