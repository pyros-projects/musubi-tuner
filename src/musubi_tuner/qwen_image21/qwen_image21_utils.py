"""Checkpoint loading and the flow-matching sampler for Qwen-Image 2.1."""

import json
import logging
import math
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from musubi_tuner.modules.int8_optimization_utils import apply_int8_convrot_monkey_patch, scan_int8_convrot
from musubi_tuner.qwen_image21.qwen_image21_model import QwenImage21Transformer2DModel
from musubi_tuner.qwen_image21.qwen_image21_vae import SPATIAL_COMPRESSION
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)

# diffusers scheduler_config.json of Qwen/Qwen-Image-2.1
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 8192
BASE_SHIFT = 0.5
MAX_SHIFT = 0.9
SHIFT_TERMINAL = 0.02


def dit_config_from_header(header: dict) -> dict:
    """Model config from tensor shapes, as Comfy's detector does; the rope split is the fixed default."""
    shape = lambda key: header[key]["shape"]  # noqa: E731
    inner_dim, in_channels = shape("img_in.weight")
    head_dim = shape("transformer_blocks.0.attn.norm_q.weight")[0]
    return dict(
        in_channels=in_channels,
        out_channels=shape("proj_out.weight")[0],
        num_layers=len({k.split(".")[1] for k in header if k.startswith("transformer_blocks.")}),
        attention_head_dim=head_dim,
        num_attention_heads=inner_dim // head_dim,
        context_in_dim=shape("txt_in.text_norm.weight")[0],
        mlp_ratio=shape("transformer_blocks.0.img_mlp.gate_up.weight")[0] // 2 // inner_dim,
    )


def load_qwen_image21_dit(
    path: str, device="cuda", dtype: torch.dtype = torch.bfloat16, axes_dims_rope=(16, 56, 56)
) -> QwenImage21Transformer2DModel:
    """Build the DiT on meta and assign weights: INT8 ConvRot Linears keep their I8 payload and F32 scales
    (frozen, forward through comfy-kitchen), everything else is cast to ``dtype``."""
    device = torch.device(device)
    with MemoryEfficientSafeOpen(path) as reader:
        keys = reader.keys()
        if "transformer_blocks.0.img_mlp.gate_up.weight" not in keys:
            raise ValueError(
                f"{path} is not a Comfy Qwen-Image 2.1 DiT (expected fused img_mlp.gate_up weights, "
                "e.g. qwen_image_2.1_int8_convrot.safetensors)"
            )
        int8_layers = scan_int8_convrot(reader)

        with torch.device("meta"):
            model = QwenImage21Transformer2DModel(**dit_config_from_header(reader.header), axes_dims_rope=axes_dims_rope)
        if int8_layers:
            apply_int8_convrot_monkey_patch(model, int8_layers)
            logger.info("Loading INT8 ConvRot Qwen-Image 2.1 DiT (%d quantized Linears)", len(int8_layers))

        quantized = {f"{name}.weight" for name in int8_layers} | {f"{name}.weight_scale" for name in int8_layers}
        sd = {}
        for key in keys:
            if key.endswith(".comfy_quant"):
                continue
            sd[key] = reader.get_tensor(key, device=device, dtype=None if key in quantized else dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded Qwen-Image 2.1 DiT from {path}")
    return model.eval().requires_grad_(False)


def calculate_mu(image_seq_len: int) -> float:
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN)
    return image_seq_len * m + BASE_SHIFT - m * BASE_IMAGE_SEQ_LEN


def get_sigmas(num_steps: int, mu: float, raw_sigmas: Optional[Sequence[float]] = None) -> torch.Tensor:
    """diffusers FlowMatchEulerDiscreteScheduler as configured for Qwen-Image 2.1: exponential time shift
    by ``mu``, stretched so the last non-zero sigma is ``SHIFT_TERMINAL``, then a trailing 0.

    ``raw_sigmas`` replaces the ``linspace(1, 1/steps, steps)`` nodes with a distilled student's own
    schedule (e.g. Viggle turbo): the nodes still get the resolution shift, but no terminal stretch,
    which would wreck a few-step student's last step."""
    sigmas = np.asarray(raw_sigmas if raw_sigmas is not None else np.linspace(1.0, 1 / num_steps, num_steps), dtype=np.float64)
    sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
    if raw_sigmas is None:
        one_minus = 1 - sigmas
        sigmas = 1 - one_minus / (one_minus[-1] / (1 - SHIFT_TERMINAL))
    return torch.tensor(np.append(sigmas, 0.0), dtype=torch.float32)


def convert_lora_to_musubi(weights_sd: dict[str, torch.Tensor], default_scale: float = 1.0) -> dict[str, torch.Tensor]:
    """Normalize a Qwen-Image 2.1 LoRA to musubi ``lora_unet_*`` keys for this DiT.

    Accepts musubi keys (returned unchanged) or ``transformer.`` / ``diffusion_model.`` prefixed keys
    with ``lora_A/lora_B`` (PEFT, diffusers) or ``lora_down/lora_up`` (+ optional ``alpha``) suffixes.
    Adapters on the unfused ``img_mlp.gate_layer`` / ``img_mlp.proj`` pair become one exact adapter on
    the fused ``img_mlp.gate_up``: concatenated downs and a block-diagonal up ([gate; up] row order).
    ``default_scale`` (alpha / rank) applies to adapters that carry no ``alpha`` tensor.
    """
    if all(k.startswith("lora_unet_") for k in weights_sd):
        return weights_sd

    suffixes = {
        ".lora_A.weight": "down",
        ".lora_down.weight": "down",
        ".lora_B.weight": "up",
        ".lora_up.weight": "up",
        ".alpha": "alpha",
    }
    modules: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in weights_sd.items():
        for prefix in ("transformer.", "diffusion_model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        suffix = next((s for s in suffixes if key.endswith(s)), None)
        if suffix is None:
            raise ValueError(f"Unsupported Qwen-Image 2.1 LoRA key: {key}")
        modules.setdefault(key[: -len(suffix)], {})[suffixes[suffix]] = value

    # bake alpha / rank into up, so every pair runs at alpha = rank and pairs can be concatenated
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for path, parts in modules.items():
        down, up = parts["down"], parts["up"]
        scale = float(parts["alpha"]) / down.shape[0] if "alpha" in parts else default_scale
        pairs[path] = (down, up if scale == 1.0 else (up.float() * scale).to(up.dtype))

    mlps = {p.rsplit(".", 1)[0] for p in pairs if p.endswith((".img_mlp.gate_layer", ".img_mlp.proj"))}
    for mlp in mlps:
        halves = [pairs.pop(f"{mlp}.gate_layer", None), pairs.pop(f"{mlp}.proj", None)]  # gate_up rows: [gate; up]
        present = [h for h in halves if h is not None]
        hidden = present[0][1].shape[0]
        down = torch.cat([h[0] for h in present], dim=0)
        up = present[0][1].new_zeros(2 * hidden, down.shape[0])
        col = 0
        for i, half in enumerate(halves):
            if half is not None:
                rank = half[0].shape[0]
                up[i * hidden : (i + 1) * hidden, col : col + rank] = half[1]
                col += rank
        pairs[f"{mlp}.gate_up"] = (down, up)

    out = {}
    for path, (down, up) in pairs.items():
        name = "lora_unet_" + path.replace(".", "_")
        out[f"{name}.lora_down.weight"] = down
        out[f"{name}.lora_up.weight"] = up
        out[f"{name}.alpha"] = torch.tensor(float(down.shape[0]))
    return out


def peft_lora_scale(path: str) -> float:
    """alpha / rank from a diffusers/PEFT file's ``lora_adapter_metadata``, 1.0 when absent (alpha = rank)."""
    with MemoryEfficientSafeOpen(path) as reader:
        raw = reader.metadata().get("lora_adapter_metadata")
    if not raw:
        return 1.0
    config = json.loads(raw)
    alpha = next((v for k, v in config.items() if k.split(".")[-1] == "lora_alpha"), None)
    rank = next((v for k, v in config.items() if k.split(".")[-1] == "r"), None)
    return float(alpha) / float(rank) if alpha and rank else 1.0


@torch.no_grad()
def denoise(
    model: QwenImage21Transformer2DModel,
    context: torch.Tensor,
    width: int,
    height: int,
    steps: int,
    generator: torch.Generator,
    negative_context: Optional[torch.Tensor] = None,
    cfg_scale: float = 1.0,
    mu: Optional[float] = None,
    raw_sigmas: Optional[Sequence[float]] = None,
    progress: Callable = lambda it, **kw: it,
) -> torch.Tensor:
    """Euler flow sampling (true CFG when ``negative_context`` is given and ``cfg_scale > 1``).

    context / negative_context: (tokens, 4096) text states. ``raw_sigmas`` (a turbo student's schedule)
    overrides ``steps``, see ``get_sigmas``. Returns normalized latents (1, 64, H/16, W/16).
    """
    device, dtype = model.device, model.dtype
    h, w = height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION
    mu = calculate_mu(h * w) if mu is None else mu
    sigmas = get_sigmas(steps, mu, raw_sigmas).tolist()
    do_cfg = negative_context is not None and cfg_scale > 1.0

    context = context.to(device, dtype).unsqueeze(0)
    if do_cfg:
        negative_context = negative_context.to(device, dtype).unsqueeze(0)
    latents = torch.randn((1, model.out_channels, h, w), generator=generator, device=device, dtype=dtype).float()
    for sigma, sigma_next in progress(list(zip(sigmas[:-1], sigmas[1:])), desc="Denoising steps"):
        x = latents.to(dtype)
        t = torch.full((1,), sigma, device=device)
        v = model(x, t, context).float()
        if do_cfg:
            v_neg = model(x, t, negative_context).float()
            v = v_neg + cfg_scale * (v - v_neg)
        latents = latents + (sigma_next - sigma) * v
    return latents
