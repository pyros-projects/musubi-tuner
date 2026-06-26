"""Convert Krea2 LoRA checkpoints from Musubi format to native ComfyUI format."""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from musubi_tuner.krea2.projector_bypass import load_projector_bypass_diff

logger = logging.getLogger(__name__)


_BLOCK_TAIL_RE = re.compile(r"^(?P<idx>\d+)_(?P<section>attn|mlp)_(?P<leaf>gate|wk|wo|wq|wv|down|up)$")


def _convert_block_tail(container: str, tail: str) -> str:
    match = _BLOCK_TAIL_RE.match(tail)
    if match is None:
        raise ValueError(f"Unsupported Krea2 block LoRA module tail: {tail}")
    return f"{container}.{match.group('idx')}.{match.group('section')}.{match.group('leaf')}"


def convert_lora_name_to_comfy_module(lora_name: str) -> str:
    """Convert a Musubi Krea2 LoRA module name to a native Krea2 module path."""
    prefix = "lora_unet_"
    if not lora_name.startswith(prefix):
        raise ValueError(f"Unsupported Krea2 LoRA name: {lora_name}")

    module_name = lora_name[len(prefix) :]

    if module_name == "first":
        return "first"
    if module_name == "last_linear":
        return "last.linear"
    if module_name == "txtfusion_projector":
        return "txtfusion.projector"

    simple_match = re.match(r"^(tmlp|txtmlp|tproj)_(\d+)$", module_name)
    if simple_match is not None:
        return f"{simple_match.group(1)}.{simple_match.group(2)}"

    if module_name.startswith("blocks_"):
        return _convert_block_tail("blocks", module_name[len("blocks_") :])
    if module_name.startswith("txtfusion_layerwise_blocks_"):
        return _convert_block_tail(
            "txtfusion.layerwise_blocks",
            module_name[len("txtfusion_layerwise_blocks_") :],
        )
    if module_name.startswith("txtfusion_refiner_blocks_"):
        return _convert_block_tail(
            "txtfusion.refiner_blocks",
            module_name[len("txtfusion_refiner_blocks_") :],
        )

    raise ValueError(f"Unsupported Krea2 LoRA module name: {lora_name}")


def convert_key_to_comfy(key: str) -> Optional[str]:
    """Convert one Musubi Krea2 LoRA tensor key to a native ComfyUI key."""
    if key.endswith(".alpha"):
        return None

    try:
        lora_name, weight_name = key.split(".", 1)
    except ValueError as exc:
        raise ValueError(f"Unexpected Krea2 LoRA key format: {key}") from exc

    if weight_name not in {"lora_down.weight", "lora_up.weight"}:
        raise ValueError(f"Unexpected Krea2 LoRA weight key: {key}")

    module_path = convert_lora_name_to_comfy_module(lora_name)
    return f"diffusion_model.{module_path}.{weight_name}"


def convert_state_dict_to_comfy(weights_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a Krea2 LoRA state dict to native ComfyUI key layout."""
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


def format_bypass_weight_token(weight: float) -> str:
    """Return a compact filesystem-safe token for a bypass weight."""
    value = f"{float(weight):g}"
    value = value.replace("-", "m").replace("+", "").replace(".", "p")
    return f"w{value}"


def default_bypass_merged_comfy_path(comfy_output_path: str | os.PathLike, bypass_weight: float) -> Path:
    output_path = Path(comfy_output_path)
    return output_path.with_name(f"{output_path.stem}.bypassed.{format_bypass_weight_token(bypass_weight)}{output_path.suffix}")


def _metadata_with_bypass_merge(
    metadata: dict[str, str] | None,
    *,
    bypass_path: str | os.PathLike,
    bypass_weight: float,
) -> dict[str, str]:
    merged_metadata = dict(metadata or {})
    merged_metadata.update(
        {
            "ss_krea2_bypass_merged": "true",
            "ss_krea2_bypass_merge_path": str(bypass_path),
            "ss_krea2_bypass_merge_weight": f"{float(bypass_weight):g}",
        }
    )
    return merged_metadata


def convert_lora_to_comfy(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike | None = None,
    verbose: bool = False,
    *,
    bypass_merge_path: str | os.PathLike | None = None,
    bypass_path: str | os.PathLike | None = None,
    bypass_weight: float = 1.0,
) -> Path:
    """Convert a saved Krea2 LoRA file to native ComfyUI format."""
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.comfy{input_path.suffix}"
    else:
        output_path = Path(output_path)
    if bypass_path is not None and bypass_merge_path is None:
        bypass_merge_path = default_bypass_merged_comfy_path(output_path, bypass_weight)
    if bypass_merge_path is not None and bypass_path is None:
        raise ValueError("Krea2 bypass-merged Comfy export requires a bypass_path.")

    if verbose:
        logger.info("Loading Krea2 LoRA from %s", input_path)
    weights_sd = load_file(str(input_path))
    converted = convert_state_dict_to_comfy(weights_sd)

    metadata = None
    with safe_open(str(input_path), framework="pt") as f:
        metadata = f.metadata()

    if verbose:
        logger.info("Saving Krea2 ComfyUI LoRA to %s (%d keys)", output_path, len(converted))
    save_file(converted, str(output_path), metadata=metadata)

    if bypass_merge_path is not None and bypass_path is not None:
        bypass_merge_path = Path(bypass_merge_path)
        bypass_diff = load_projector_bypass_diff(bypass_path) * float(bypass_weight)
        bypass_merged = dict(converted)
        bypass_merged["diffusion_model.txtfusion.projector.diff"] = bypass_diff
        bypass_metadata = _metadata_with_bypass_merge(metadata, bypass_path=bypass_path, bypass_weight=bypass_weight)
        if verbose:
            logger.info("Saving Krea2 bypass-merged ComfyUI LoRA to %s (%d keys)", bypass_merge_path, len(bypass_merged))
        save_file(bypass_merged, str(bypass_merge_path), metadata=bypass_metadata)

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Krea2 LoRA from Musubi format to native ComfyUI format")
    parser.add_argument("input", type=str, help="Path to the input Krea2 LoRA checkpoint")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to save the converted LoRA (default: <input>.comfy.safetensors)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print conversion details")
    parser.add_argument("--bypass", type=str, default=None, help="Optional Krea2 projector bypass diff to merge into a second ComfyUI output")
    parser.add_argument(
        "--bypass-weight",
        "--bypass_weight",
        dest="bypass_weight",
        type=float,
        default=1.0,
        help="Multiplier for --bypass when writing the bypass-merged ComfyUI output.",
    )
    parser.add_argument(
        "--bypass-merge-output",
        "--bypass_merge_output",
        dest="bypass_merge_output",
        type=str,
        default=None,
        help="Path for the bypass-merged ComfyUI output (default: <output>.bypassed.<weight>.safetensors).",
    )
    args = parser.parse_args()

    convert_lora_to_comfy(
        args.input,
        args.output,
        verbose=args.verbose,
        bypass_merge_path=args.bypass_merge_output,
        bypass_path=args.bypass,
        bypass_weight=args.bypass_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
