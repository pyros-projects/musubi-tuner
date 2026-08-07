"""Convert MiniMax H3 LoRA from musubi (kohya) format to ComfyUI format.

Training format (musubi/kohya):
  lora_unet_blocks_12_mlp_fc1.lora_down.weight
  lora_unet_blocks_12_mlp_fc1.lora_up.weight
  lora_unet_blocks_12_mlp_fc1.alpha

ComfyUI format (what ai-toolkit ships and ComfyUI loads natively):
  diffusion_model.blocks.12.mlp.fc1.lora_A.weight
  diffusion_model.blocks.12.mlp.fc1.lora_B.weight
  (no alpha keys — the kohya alpha/rank scale is folded into lora_B)
"""

from __future__ import annotations

import argparse
import logging
import re

import torch
from safetensors.torch import save_file

from musubi_tuner.utils import safetensors_utils

logger = logging.getLogger(__name__)

KOHYA_PREFIX = "lora_unet_"

# Module-path atoms whose internal underscores must survive the underscore->dot
# conversion. Ordered: composite atoms first, block prefixes last.
_ATOM_REPLACEMENTS = (
    ("_attn_qkv_proj", ".attn.qkv_proj"),
    ("_attn_out_proj", ".attn.out_proj"),
    ("_mlp_fc1", ".mlp.fc1"),
    ("_mlp_fc2", ".mlp.fc2"),
    ("_adaln_proj_linear", ".adaln_proj.linear"),
    ("token_refiner_blocks_", "token_refiner.blocks."),
    ("blocks_", "blocks."),
)

_KNOWN_MODULE_RE = re.compile(
    r"^(?:token_refiner\.)?blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.fc[12]|adaln_proj\.linear)$"
)


def convert_module_name_to_comfy(kohya_module: str) -> str:
    """Map a kohya-mangled H3 module name to its dotted model path."""
    if not kohya_module.startswith(KOHYA_PREFIX):
        raise ValueError(f"Not a musubi H3 LoRA module key: {kohya_module}")
    name = kohya_module[len(KOHYA_PREFIX) :]
    for src, dst in _ATOM_REPLACEMENTS:
        name = name.replace(src, dst)
    if not _KNOWN_MODULE_RE.match(name):
        logger.warning("Unrecognized H3 LoRA module path after conversion: %s", name)
    return name


def convert_lora_to_comfy(src_path: str, dst_path: str) -> int:
    """Convert a musubi H3 LoRA safetensors file to ComfyUI format.

    Returns the number of LoRA modules converted.
    """
    with safetensors_utils.MemoryEfficientSafeOpen(src_path) as f:
        metadata = f.metadata() or {}
        sd = {k: f.get_tensor(k) for k in f.keys()}

    modules: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in sd.items():
        module, _, suffix = key.partition(".")
        if suffix not in ("lora_down.weight", "lora_up.weight", "alpha"):
            raise ValueError(f"Unexpected key in LoRA checkpoint: {key}")
        modules.setdefault(module, {})[suffix] = tensor

    out_sd: dict[str, torch.Tensor] = {}
    for module, tensors in sorted(modules.items()):
        down = tensors.get("lora_down.weight")
        up = tensors.get("lora_up.weight")
        if down is None or up is None:
            raise ValueError(f"Incomplete LoRA module {module}: {sorted(tensors)}")
        rank = down.shape[0]
        alpha = tensors.get("alpha")
        scale = (alpha.float().item() / rank) if alpha is not None else 1.0
        if scale != 1.0:
            up = (up.float() * scale).to(up.dtype)
        comfy_module = f"diffusion_model.{convert_module_name_to_comfy(module)}"
        out_sd[f"{comfy_module}.lora_A.weight"] = down
        out_sd[f"{comfy_module}.lora_B.weight"] = up

    metadata = dict(metadata)
    metadata["ss_comfy_converted"] = "minimax_h3"
    save_file(out_sd, dst_path, metadata=metadata)
    return len(modules)


def convert_diffusers_lora_to_comfy(src_path: str, dst_path: str) -> int:
    """Convert a diffusers-PEFT H3 LoRA (e.g. lightx2v turbo) to ComfyUI format.

    Split to_q/to_k/to_v adapters are fused into one block-diagonal pair on
    qkv_proj (rank = sum of the parts), the diffusers SwiGLU half-swap on fc1 is
    undone, and the external reference alpha (8) is folded into lora_B so the
    file behaves correctly at strength 1.0 in any stock loader.
    """
    from musubi_tuner.minimax_h3 import sampling_lora_overlay

    with safetensors_utils.MemoryEfficientSafeOpen(src_path) as f:
        metadata = f.metadata() or {}
        sd = {k: f.get_tensor(k) for k in f.keys()}

    modules = sampling_lora_overlay.normalize_overlay_state_dict(sd)
    out_sd: dict[str, torch.Tensor] = {}
    for path, entries in sorted(modules.items()):
        entries = sorted(entries, key=lambda e: e["offset"] or 0)
        scaled = []
        for e in entries:
            scale = (e["alpha"] / e["down"].shape[0]) if e["alpha"] is not None else 1.0
            up = (e["up"].float() * scale).to(e["up"].dtype)
            scaled.append((e["down"], up, e["offset"] or 0))
        if len(scaled) == 1 and entries[0]["offset"] is None:
            down, up = scaled[0][0], scaled[0][1]
        else:
            out_features = max(off + up.shape[0] for _, up, off in scaled)
            total_rank = sum(d.shape[0] for d, _, _ in scaled)
            down = torch.cat([d for d, _, _ in scaled], dim=0)
            up = torch.zeros(out_features, total_rank, dtype=scaled[0][1].dtype)
            col = 0
            for d, u, off in scaled:
                up[off : off + u.shape[0], col : col + d.shape[0]] = u
                col += d.shape[0]
        comfy_module = f"diffusion_model.{path}"
        out_sd[f"{comfy_module}.lora_A.weight"] = down
        out_sd[f"{comfy_module}.lora_B.weight"] = up

    metadata = dict(metadata)
    metadata["ss_comfy_converted"] = "minimax_h3_diffusers"
    save_file(out_sd, dst_path, metadata=metadata)
    return len(modules)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a MiniMax H3 LoRA (musubi or diffusers-PEFT) to ComfyUI format")
    parser.add_argument("input", help="LoRA .safetensors (musubi/kohya or diffusers-PEFT keys, auto-detected)")
    parser.add_argument("output", nargs="?", default=None, help="output path (default: <input>.comfy.safetensors)")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = re.sub(r"\.safetensors$", "", args.input) + ".comfy.safetensors"

    with safetensors_utils.MemoryEfficientSafeOpen(args.input) as f:
        is_diffusers = any(key.endswith(".default.weight") for key in f.keys())
    if is_diffusers:
        count = convert_diffusers_lora_to_comfy(args.input, output)
    else:
        count = convert_lora_to_comfy(args.input, output)
    print(f"Converted {count} LoRA modules -> {output}")


if __name__ == "__main__":
    main()
