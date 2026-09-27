"""Reduce the rank of an LTX-2 LoRA by factored SVD truncation.

Built for the huge official distilled LoRAs (e.g. the 8.9GB rank-450
ltx-2.5 file): per module, the delta B@A is decomposed via the QR-factored
SVD (never materializing more than one module at a time), singular values
are kept until the requested Frobenius-energy fraction is reached, capped
by a rank ceiling, with an optional condition safeguard dropping tiny
singular values relative to the largest.

Input/output format: comfy-style ``diffusion_model.<module>.lora_A.weight`` /
``lora_B.weight`` pairs. ``.alpha`` tensors, when present, are baked into the
delta (alpha/rank scaling) and omitted from the output, so the reduced file
reproduces the same effective delta at multiplier 1.0.

Usage:
    python ltx2_reduce_lora.py in.safetensors --energy 0.90 --max-rank 72
"""

import argparse
import json
import logging
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def truncated_pair(
    down: torch.Tensor,
    up: torch.Tensor,
    *,
    energy: float,
    max_rank: int,
    cond_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """Return (down', up', kept_rank, kept_energy) with up'@down' ~= up@down.

    ``down`` is lora_A [r, in], ``up`` is lora_B [out, r]. Factored SVD:
    QR both thin factors, SVD the small [r, r] core.
    """
    down32 = down.to(torch.float32)
    up32 = up.to(torch.float32)

    q_up, r_up = torch.linalg.qr(up32, mode="reduced")  # [out, r], [r, r]
    q_down, r_down = torch.linalg.qr(down32.T, mode="reduced")  # [in, r], [r, r]
    core = r_up @ r_down.T  # [r, r]
    u, s, vt = torch.linalg.svd(core, full_matrices=False)

    total = float((s * s).sum())
    if total <= 0.0:
        return down32[:1].to(down.dtype), torch.zeros_like(up32[:, :1]).to(up.dtype), 1, 1.0

    cumulative = torch.cumsum(s * s, dim=0) / total
    k = int(torch.searchsorted(cumulative, energy).item()) + 1
    k = max(1, min(k, max_rank, s.numel()))
    # Condition safeguard: never keep singular values tiny relative to s[0].
    if cond_epsilon > 0.0:
        significant = int((s >= s[0] * cond_epsilon).sum().item())
        k = max(1, min(k, significant))

    kept_energy = float(cumulative[k - 1])
    sqrt_s = torch.sqrt(s[:k])
    new_up = (q_up @ u[:, :k]) * sqrt_s  # [out, k]
    new_down = (sqrt_s[:, None] * vt[:k] @ q_down.T)  # [k, in]
    return new_down.to(down.dtype), new_up.to(up.dtype), k, kept_energy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lora", type=Path, help="Input LoRA safetensors (comfy lora_A/lora_B format)")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: derives _fro<E>_ceil<R> name)")
    parser.add_argument("--energy", type=float, default=0.90, help="Frobenius energy fraction to keep (default 0.90)")
    parser.add_argument("--max-rank", type=int, default=72, help="Rank ceiling per module (default 72)")
    parser.add_argument("--cond-epsilon", type=float, default=1e-4, help="Drop singular values < eps * s_max (0 disables)")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.energy <= 1.0:
        parser.error(f"--energy must be in (0, 1], got {args.energy}")
    if args.max_rank < 1:
        parser.error(f"--max-rank must be >= 1, got {args.max_rank}")

    output = args.output
    if output is None:
        pct = int(round(args.energy * 100))
        output = args.lora.with_name(f"{args.lora.stem}_fro{pct}_ceil{args.max_rank}{args.lora.suffix}")
    if output.exists() and not args.overwrite:
        parser.error(f"Output exists: {output} (use --overwrite)")

    out_tensors: dict[str, torch.Tensor] = {}
    report: dict[str, dict] = {}
    with safe_open(str(args.lora), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        a_keys = sorted(k for k in keys if k.endswith(".lora_A.weight"))
        if not a_keys:
            raise SystemExit("No .lora_A.weight tensors found — is this a comfy-format LoRA?")
        consumed = set()
        for a_key in a_keys:
            prefix = a_key[: -len(".lora_A.weight")]
            b_key = f"{prefix}.lora_B.weight"
            alpha_key = f"{prefix}.alpha"
            if b_key not in keys:
                raise SystemExit(f"Missing pair for {a_key}")
            down = f.get_tensor(a_key)
            up = f.get_tensor(b_key)
            if alpha_key in keys:
                alpha = float(f.get_tensor(alpha_key))
                # Bake alpha/rank scaling into the delta; the output carries no alpha.
                up = up * (alpha / down.shape[0])
                consumed.add(alpha_key)
            new_down, new_up, kept, kept_energy = truncated_pair(
                down, up, energy=args.energy, max_rank=args.max_rank, cond_epsilon=args.cond_epsilon
            )
            out_tensors[a_key] = new_down.contiguous()
            out_tensors[b_key] = new_up.contiguous()
            report[prefix] = {"rank_in": down.shape[0], "rank_out": kept, "energy": round(kept_energy, 4)}
            consumed.update((a_key, b_key))

        passthrough = keys - consumed
        for key in sorted(passthrough):
            out_tensors[key] = f.get_tensor(key)
        if passthrough:
            logger.info("Passing through %d non-LoRA tensors unchanged", len(passthrough))

    ranks = [r["rank_out"] for r in report.values()]
    energies = [r["energy"] for r in report.values()]
    logger.info(
        "Reduced %d modules: rank min/median/max = %d/%d/%d, worst kept energy = %.3f",
        len(report), min(ranks), sorted(ranks)[len(ranks) // 2], max(ranks), min(energies),
    )

    size = sum(t.numel() * t.element_size() for t in out_tensors.values())
    save_file(out_tensors, str(output), metadata={"reduced_from": args.lora.name,
                                                  "energy": str(args.energy), "max_rank": str(args.max_rank)})
    logger.info("Saved %s (%.2f GB)", output, size / 1e9)

    if args.report:
        args.report.write_text(json.dumps(report, indent=1))
        logger.info("Report: %s", args.report)


if __name__ == "__main__":
    main()
