import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

SUPPORTED_PROJECTOR_DIFF_KEYS = (
    "diffusion_model.txtfusion.projector.diff",
    "txtfusion.projector.diff",
    "diffusion_model.txtfusion.projector.weight.diff",
    "txtfusion.projector.weight.diff",
)


def load_projector_bypass_diff(path: str | Path) -> torch.Tensor:
    state_dict = load_file(str(path))
    for key in SUPPORTED_PROJECTOR_DIFF_KEYS:
        if key in state_dict:
            return state_dict[key].detach().to(torch.float32)

    supported = ", ".join(SUPPORTED_PROJECTOR_DIFF_KEYS)
    raise ValueError(f"No supported Krea2 projector diff found in {path}. Expected one of: {supported}")


def apply_projector_bypass(model: torch.nn.Module, diff: torch.Tensor, weight: float, source_path: str | Path | None = None) -> None:
    try:
        projector_weight = model.txtfusion.projector.weight
    except AttributeError as exc:
        raise ValueError("Krea2 projector bypass requires model.txtfusion.projector.weight") from exc

    if tuple(projector_weight.shape) != tuple(diff.shape):
        raise ValueError(
            f"Krea2 projector bypass shape mismatch: projector weight shape {tuple(projector_weight.shape)} "
            f"does not match diff shape {tuple(diff.shape)}"
        )

    with torch.no_grad():
        projector_weight.add_(diff.to(device=projector_weight.device, dtype=projector_weight.dtype) * float(weight))

    source = f" from {source_path}" if source_path is not None else ""
    logger.info(f"Applied Krea2 projector bypass{source} with weight {float(weight):g}")


def apply_projector_bypass_from_file(model: torch.nn.Module, path: str | Path, weight: float) -> None:
    diff = load_projector_bypass_diff(path)
    apply_projector_bypass(model, diff, weight, source_path=path)
