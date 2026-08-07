"""Runtime LoRA overlays for MiniMax H3 preview sampling.

Lets previews run with auxiliary frozen LoRAs stacked on the training model —
e.g. the community MiniMax-H3 Turbo LoRA for 4-step previews — without merging
anything into weights. Overlays add ``scale * up @ down @ x`` in activation
space (the only possibility for INT8 base layers, and lossless for tiny
distill deltas that a bf16 merge would round away) and are detached after the
sampling round, so training is untouched.

Accepts the three H3 LoRA key formats in the wild: musubi/kohya
(``lora_unet_blocks_0_mlp_fc1.lora_down.weight``), ComfyUI/ai-toolkit
(``diffusion_model.blocks.0.mlp.fc1.lora_A.weight``) and the bare Turbo-node
layout (``blocks.0.attn.qkv_proj.lora_A.weight``).

AdaLN overlays need special handling on pruned (curve-table) bases: the LoRA
delta lives in the full checkpoint's 2688-dim ``silu(t_emb)`` space, which a
pruned model never materializes. Following the Turbo node's approach, a
precomputed ``silu(t_emb)`` grid is interpolated at the sampled timesteps
(``update_time_state``, called by the preview sampler before every model call)
and the low-rank delta is added to the adaln projection output.
"""

from __future__ import annotations

import logging
import math
import re

import torch
import torch.nn.functional as F

from musubi_tuner.minimax_h3.convert_lora_to_comfy import KOHYA_PREFIX, convert_module_name_to_comfy
from musubi_tuner.minimax_h3.model import time_shift_sigma
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)

_ORIG_FORWARD_ATTR = "_h3_sampling_overlay_orig_forward"
_ADAPTERS_ATTR = "_h3_sampling_overlay_adapters"
_SHARED_ATTR = "_h3_sampling_overlay_shared"

_PAIR_RE = re.compile(r"^(?:diffusion_model\.)?(.+)\.(lora_A|lora_B|lora_down|lora_up)\.weight$")
_ALPHA_RE = re.compile(r"^(?:diffusion_model\.)?(.+)\.alpha$")


def load_overlay_file(path: str) -> dict[str, torch.Tensor]:
    with MemoryEfficientSafeOpen(path) as reader:
        return {key: reader.get_tensor(key) for key in reader.keys()}


def load_temb_grid(path: str) -> torch.Tensor:
    """Load the silu(t_emb) grid shipped with the Turbo node ([grid, 2688] fp32)."""
    with MemoryEfficientSafeOpen(path) as reader:
        keys = reader.keys()
        key = "silu_t_emb_grid" if "silu_t_emb_grid" in keys else keys[0]
        grid = reader.get_tensor(key).float()
    if grid.ndim != 2 or grid.shape[0] < 2:
        raise ValueError(f"Invalid silu(t_emb) grid shape {tuple(grid.shape)} in {path}")
    return grid


def normalize_overlay_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Map any supported LoRA key format to {module_path: {down, up, alpha}}."""

    modules: dict[str, dict] = {}

    def entry(path: str) -> dict:
        return modules.setdefault(path, {"down": None, "up": None, "alpha": None})

    for key, tensor in sd.items():
        if key.startswith(KOHYA_PREFIX):
            module_key, _, suffix = key.partition(".")
            path = convert_module_name_to_comfy(module_key)
            if suffix == "lora_down.weight":
                entry(path)["down"] = tensor
            elif suffix == "lora_up.weight":
                entry(path)["up"] = tensor
            elif suffix == "alpha":
                entry(path)["alpha"] = float(tensor)
            else:
                raise ValueError(f"Unexpected musubi LoRA key: {key}")
            continue
        pair = _PAIR_RE.match(key)
        if pair:
            path, kind = pair.group(1), pair.group(2)
            if kind in ("lora_A", "lora_down"):
                entry(path)["down"] = tensor
            else:
                entry(path)["up"] = tensor
            continue
        alpha = _ALPHA_RE.match(key)
        if alpha:
            entry(alpha.group(1))["alpha"] = float(tensor)
            continue
        raise ValueError(f"Unrecognized H3 LoRA overlay key: {key}")

    for path, tensors in modules.items():
        if tensors["down"] is None or tensors["up"] is None:
            raise ValueError(f"Incomplete LoRA overlay module {path}")
    return modules


def _module_scale(tensors: dict, strength: float) -> float:
    alpha = tensors["alpha"]
    rank = tensors["down"].shape[0]
    return strength * (alpha / rank if alpha is not None else 1.0)


def _overlay_linear_forward(self, x, *args, **kwargs):
    out = getattr(self, _ORIG_FORWARD_ATTR)(x, *args, **kwargs)
    for adapter in getattr(self, _ADAPTERS_ATTR):
        down, up = adapter["down"], adapter["up"]
        if down.device != x.device:
            down = adapter["down"] = down.to(x.device)
            up = adapter["up"] = up.to(x.device)
        delta = F.linear(F.linear(x.to(down.dtype), down), up)
        out = out + adapter["scale"] * delta.to(dtype=out.dtype, device=out.device)
    return out


def _overlay_adaln_curve_forward(self, time_embedding):
    # Curve-mode AdalnProj.forward with the overlay delta re-injected in the full
    # silu(t_emb) space (mirrors musubi_tuner.minimax_h3.model.AdalnProj).
    projected = self.linear(time_embedding)
    shared = getattr(self, _SHARED_ATTR)
    silu_temb = shared.get("silu_temb")
    if silu_temb is not None:
        for adapter in getattr(self, _ADAPTERS_ATTR):
            down, up = adapter["down"], adapter["up"]
            if down.device != projected.device:
                down = adapter["down"] = down.to(projected.device)
                up = adapter["up"] = up.to(projected.device)
            rows = silu_temb.to(device=down.device, dtype=down.dtype)
            delta = F.linear(F.linear(rows, down), up)
            projected = projected + adapter["scale"] * delta.to(projected.dtype)
    projected = projected.view(projected.shape[0] * self.modalities, self.expand * self.hidden)
    return projected.chunk(self.expand, dim=-1)


def _attach_adapter(module: torch.nn.Module, forward_impl, down: torch.Tensor, up: torch.Tensor, scale: float, dtype: torch.dtype):
    if not hasattr(module, _ORIG_FORWARD_ATTR):
        setattr(module, _ORIG_FORWARD_ATTR, module.forward)
        setattr(module, _ADAPTERS_ATTR, [])
        module.forward = forward_impl.__get__(module, module.__class__)
    getattr(module, _ADAPTERS_ATTR).append(
        {"down": down.detach().to(dtype), "up": up.detach().to(dtype), "scale": float(scale)}
    )


def attach_sampling_lora_overlays(
    model: torch.nn.Module,
    overlays: list[tuple[dict[str, dict], float]],
    *,
    temb_grid: torch.Tensor | None = None,
) -> dict[str, int | list[str]]:
    """Attach overlay adapters for each (normalized_modules, strength) pair."""

    stats = {"backbone": 0, "adaln_grid": 0, "skipped": []}
    use_curves = bool(getattr(model, "use_adaln_curves", False))
    shared = {"silu_temb": None, "grid": temb_grid}

    for modules, strength in overlays:
        for path, tensors in sorted(modules.items()):
            scale = _module_scale(tensors, strength)
            is_adaln = ".adaln_proj" in path or path.startswith("final_layer.adaln_proj")
            if is_adaln and use_curves:
                if temb_grid is None:
                    stats["skipped"].append(path)
                    continue
                adaln_path = path[: -len(".linear")] if path.endswith(".linear") else path
                adaln_module = model.get_submodule(adaln_path)
                if getattr(adaln_module, "apply_silu", False):
                    raise ValueError(f"Expected curve-mode AdalnProj without silu at {adaln_path}")
                if not hasattr(adaln_module, _SHARED_ATTR):
                    setattr(adaln_module, _SHARED_ATTR, shared)
                _attach_adapter(adaln_module, _overlay_adaln_curve_forward, tensors["down"], tensors["up"], scale, torch.float32)
                stats["adaln_grid"] += 1
            else:
                try:
                    module = model.get_submodule(path)
                except AttributeError:
                    stats["skipped"].append(path)
                    continue
                weight = getattr(module, "weight", None)
                is_int8 = isinstance(weight, torch.Tensor) and weight.dtype == torch.int8
                # fp32 math over quantized bases (sibling-repo precedent); source dtype otherwise
                dtype = torch.float32 if is_int8 else tensors["down"].dtype
                _attach_adapter(module, _overlay_linear_forward, tensors["down"], tensors["up"], scale, dtype)
                stats["backbone"] += 1

    if stats["adaln_grid"]:
        setattr(model, _SHARED_ATTR, shared)
    if stats["skipped"]:
        preview = ", ".join(stats["skipped"][:4])
        reason = "no silu(t_emb) grid provided" if use_curves and temb_grid is None else "module not found on this base"
        logger.warning(
            "Sampling LoRA overlay skipped %d modules (%s): %s%s",
            len(stats["skipped"]),
            reason,
            preview,
            ", ..." if len(stats["skipped"]) > 4 else "",
        )
    return stats


def update_time_state(model: torch.nn.Module, sigma_video: torch.Tensor | float) -> None:
    """Interpolate silu(t_emb) rows for the model's (t_video, t_audio) at this step.

    Must mirror ``MiniMaxH3Model._forward_single``: ``sorted({1-sigma_v, 1-sigma_a})``
    with ``sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)`` so overlay rows
    align with the model's unique-time rows.
    """

    shared = getattr(model, _SHARED_ATTR, None)
    if shared is None or shared.get("grid") is None:
        return
    sigma_v = min(max(float(torch.as_tensor(sigma_video).flatten()[0]), 1e-6), 1.0)
    sigma_a = float(time_shift_sigma(torch.tensor(sigma_v), model.sigma_shift_video, model.sigma_shift_audio))
    unique_times = sorted({1.0 - sigma_v, 1.0 - sigma_a})

    grid = shared["grid"]
    n = grid.shape[0]
    rows = []
    for t in unique_times:
        position = min(max(t, 0.0), 1.0) * (n - 1)
        lower = min(int(math.floor(position)), n - 2)
        rows.append(torch.lerp(grid[lower], grid[lower + 1], position - lower))
    shared["silu_temb"] = torch.stack(rows)


def clear_sampling_lora_overlays(model: torch.nn.Module) -> int:
    cleared = 0
    for module in model.modules():
        if hasattr(module, _ORIG_FORWARD_ATTR):
            module.forward = getattr(module, _ORIG_FORWARD_ATTR)
            delattr(module, _ORIG_FORWARD_ATTR)
            delattr(module, _ADAPTERS_ATTR)
            cleared += 1
        if hasattr(module, _SHARED_ATTR):
            delattr(module, _SHARED_ATTR)
    return cleared
