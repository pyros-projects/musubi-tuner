"""Post-hoc EMA over a run's saved LoRA checkpoints.

Computes the true EMA of the low-rank deltas: for each module the weighted sum
of K checkpoint deltas is a rank<=K*r matrix, held in factored form
(concatenated sqrt-weighted factors) and truncated back to the original rank
with an exact factored SVD (QR of both factor stacks + a small core SVD — the
full delta is never materialized). The captured-energy fraction is reported so
the truncation is honest; on real runs the trajectory shares one low-rank
subspace and rank-r keeps >99.5% of the energy.

Usage:
    python -m musubi_tuner.minimax_h3.posthoc_ema RUN_DIR [--beta 0.9 0.75 0]

beta = 0 is the uniform checkpoint average; otherwise weights are
``(1-beta) * beta^(K-1-k)`` (newest checkpoint heaviest). Writes
``<run>-ema_<tag>.safetensors`` next to the checkpoints, plus a ComfyUI-format
twin unless --no_comfy is given.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def load_lora_checkpoint(path: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, float]], dict]:
    modules: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    with safe_open(str(path), "pt") as reader:
        metadata = dict(reader.metadata() or {})
        for base in sorted({key.rsplit(".", 2)[0] for key in reader.keys() if key.endswith(".lora_down.weight")}):
            down = reader.get_tensor(f"{base}.lora_down.weight").float()
            up = reader.get_tensor(f"{base}.lora_up.weight").float()
            try:
                alpha = float(reader.get_tensor(f"{base}.alpha"))
            except Exception:
                alpha = float(down.shape[0])
            modules[base] = (down, up, alpha)
    return modules, metadata


def truncated_pair(stack_down: torch.Tensor, stack_up: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Exact SVD of ``stack_up @ stack_down`` from factored form; rank-r pair + energy kept."""
    qb, rb = torch.linalg.qr(stack_up)
    qa, ra = torch.linalg.qr(stack_down.T)
    u, s, vh = torch.linalg.svd(rb @ ra.T)
    energy = float((s[:rank] ** 2).sum() / (s**2).sum().clamp_min(1e-12))
    root = s[:rank].sqrt()
    down = root[:, None] * (vh[:rank] @ qa.T)
    up = (qb @ u[:, :rank]) * root
    return down, up, energy


def ema_weights(count: int, beta: float) -> tuple[list[float], str]:
    if beta == 0.0:
        return [1.0 / count] * count, "uniform"
    weights = [(1 - beta) * beta ** (count - 1 - index) for index in range(count)]
    weights[0] += beta**count  # fold the tail so weights sum to 1
    return weights, f"b{int(round(beta * 100)):02d}"


def find_step_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints = [path for path in run_dir.glob("*step[0-9]*.safetensors") if ".comfy." not in path.name and "-ema_" not in path.name]
    return sorted(checkpoints, key=lambda path: int(re.search(r"step(\d+)", path.name).group(1)))


def compute_ema(checkpoints: list[Path], beta: float) -> tuple[dict[str, torch.Tensor], dict, str, dict]:
    weights, tag = ema_weights(len(checkpoints), beta)
    stacks: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    alphas: dict[str, float] = {}
    metadata: dict = {}
    for checkpoint, weight in zip(checkpoints, weights):
        modules, metadata = load_lora_checkpoint(checkpoint)
        for base, (down, up, alpha) in modules.items():
            scale_root = (weight * alpha / down.shape[0]) ** 0.5
            downs, ups = stacks.setdefault(base, ([], []))
            downs.append(scale_root * down)
            ups.append(scale_root * up)
            alphas[base] = alpha

    state_dict: dict[str, torch.Tensor] = {}
    energies: list[float] = []
    total_dw = 0.0
    for base, (downs, ups) in stacks.items():
        rank = downs[0].shape[0]
        down, up, energy = truncated_pair(torch.cat(downs, dim=0), torch.cat(ups, dim=1), rank)
        energies.append(energy)
        alpha = alphas[base]
        # the factored pair carries the delta at scale 1; restore the alpha/rank convention
        unscale = (down.shape[0] / alpha) ** 0.5
        state_dict[f"{base}.lora_down.weight"] = (down * unscale).to(torch.bfloat16).contiguous()
        state_dict[f"{base}.lora_up.weight"] = (up * unscale).to(torch.bfloat16).contiguous()
        state_dict[f"{base}.alpha"] = torch.tensor(alpha)
        total_dw += float(torch.linalg.matrix_norm(up @ down).item())

    energy_tensor = torch.tensor(energies)
    stats = {"dw": total_dw, "energy_min": float(energy_tensor.min()), "energy_mean": float(energy_tensor.mean())}
    return state_dict, metadata, tag, stats


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Post-hoc EMA over saved H3 LoRA step checkpoints")
    parser.add_argument("run_dir", type=str, help="run output directory containing *-stepNNNNNNNN.safetensors")
    parser.add_argument("--beta", type=float, nargs="+", default=[0.9], help="EMA decays; 0 = uniform average")
    parser.add_argument("--no_comfy", action="store_true", help="skip writing the ComfyUI-format twin")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoints = find_step_checkpoints(run_dir)
    if len(checkpoints) < 2:
        raise SystemExit(f"need at least 2 step checkpoints in {run_dir}")
    first = re.search(r"step(\d+)", checkpoints[0].name).group(1)
    last = re.search(r"step(\d+)", checkpoints[-1].name).group(1)
    logger.info("%s: %d checkpoints, steps %s..%s", run_dir.name, len(checkpoints), first, last)

    for beta in args.beta:
        if not 0.0 <= beta < 1.0:
            raise SystemExit(f"--beta must be in [0, 1): {beta}")
        state_dict, metadata, tag, stats = compute_ema(checkpoints, beta)
        metadata["ss_posthoc_ema"] = f"beta={beta} over {len(checkpoints)} checkpoints"
        output = run_dir / f"{run_dir.name}-ema_{tag}.safetensors"
        save_file(state_dict, str(output), metadata=metadata)
        logger.info(
            "%s: dw=%.1f energy min/mean=%.4f/%.4f -> %s",
            tag, stats["dw"], stats["energy_min"], stats["energy_mean"], output.name,
        )
        if not args.no_comfy:
            from musubi_tuner.minimax_h3.convert_lora_to_comfy import convert_lora_to_comfy

            comfy_output = output.with_name(output.stem + ".comfy.safetensors")
            converted = convert_lora_to_comfy(str(output), str(comfy_output))
            logger.info("Converted %d LoRA modules -> %s", converted, comfy_output.name)


if __name__ == "__main__":
    main()
