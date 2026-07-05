#!/usr/bin/env python3
"""Extract an LTX-2 LoRA from the difference between two full checkpoints.

The checkpoint pair may include transformer, VAE, audio VAE, vocoder, and other
components. This tool intentionally extracts only selected 2D weights from
``model.diffusion_model`` by default, so text/autoencoder/audio-decoder payloads
do not leak into a LoRA adapter by accident.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

PRESET_PATTERNS: dict[str, list[str] | None] = {
    "t2v": [
        r".*\.to_k$",
        r".*\.to_q$",
        r".*\.to_v$",
        r".*\.to_out\.0$",
    ],
    "video_attn": [
        r".*\.attn1\.to_k$",
        r".*\.attn1\.to_q$",
        r".*\.attn1\.to_v$",
        r".*\.attn1\.to_out\.0$",
        r".*\.attn2\.to_k$",
        r".*\.attn2\.to_q$",
        r".*\.attn2\.to_v$",
        r".*\.attn2\.to_out\.0$",
    ],
    "video_attn_ffn": [
        r".*\.attn1\.to_k$",
        r".*\.attn1\.to_q$",
        r".*\.attn1\.to_v$",
        r".*\.attn1\.to_out\.0$",
        r".*\.attn2\.to_k$",
        r".*\.attn2\.to_q$",
        r".*\.attn2\.to_v$",
        r".*\.attn2\.to_out\.0$",
        r".*\.ff\.net\.0\.proj$",
        r".*\.ff\.net\.2$",
    ],
    "v2v": [
        r".*\.to_k$",
        r".*\.to_q$",
        r".*\.to_v$",
        r".*\.to_out\.0$",
        r".*\.ff\.net\.0\.proj$",
        r".*\.ff\.net\.2$",
        r".*\.audio_ff\.net\.0\.proj$",
        r".*\.audio_ff\.net\.2$",
    ],
    "full": None,
}


@dataclass(frozen=True)
class ExtractionGroup:
    source_key: str
    module: str
    shape: tuple[int, int]


@dataclass(frozen=True)
class ExtractionReportItem:
    source_key: str
    module: str
    tensor_shape: tuple[int, int]
    status: str
    actual_rank: int
    max_rank: int
    retained_energy: float
    relative_frobenius_error: float
    total_energy: float
    discarded_energy: float
    svd_method: str


class TensorSource:
    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"Expected a safetensors file, got: {path}")
        self.path = path
        with safe_open(self.path, framework="pt", device="cpu") as handle:
            self._keys = list(handle.keys())
            self._shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in self._keys}

    def keys(self) -> list[str]:
        return list(self._keys)

    def has(self, key: str) -> bool:
        return key in self._shapes

    def shape(self, key: str) -> tuple[int, ...]:
        return self._shapes[key]

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self._shapes:
            raise KeyError(f"Tensor not found in {self.path}: {key}")
        with safe_open(self.path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
            scale_key = self._scale_key_for_weight(key)
            if scale_key is None:
                return tensor
            scale = handle.get_tensor(scale_key).to(torch.float32)
            return self._apply_weight_scale(tensor, scale)

    def _scale_key_for_weight(self, key: str) -> str | None:
        if not key.endswith(".weight"):
            return None
        for scale_key in (key.replace(".weight", ".weight_scale"), key.replace(".weight", ".scale_weight")):
            if scale_key in self._shapes:
                return scale_key
        return None

    @staticmethod
    def _apply_weight_scale(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(torch.float32)
        if scale.ndim < 3:
            return tensor * scale

        if tensor.ndim != 2:
            raise ValueError(f"Block-scaled weight must be 2D, got shape {tuple(tensor.shape)}")
        out_features, num_blocks, _ = scale.shape
        if tensor.shape[0] != out_features:
            raise ValueError(
                f"Block-scaled weight output dimension mismatch: weight={tuple(tensor.shape)} scale={tuple(scale.shape)}"
            )
        if tensor.shape[1] % num_blocks != 0:
            raise ValueError(f"Block-scaled weight input dimension mismatch: weight={tuple(tensor.shape)} scale={tuple(scale.shape)}")

        return (tensor.contiguous().view(out_features, num_blocks, -1) * scale).view(tensor.shape)


def _compile_regexes(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def _matches_any(text: str, regexes: list[re.Pattern[str]]) -> bool:
    return any(regex.fullmatch(text) or regex.search(text) for regex in regexes)


def _module_from_key(key: str) -> str:
    prefix = "model.diffusion_model."
    suffix = ".weight"
    if not key.startswith(prefix) or not key.endswith(suffix):
        raise ValueError(f"Not an LTX-2 diffusion_model weight key: {key}")
    return key[len(prefix) : -len(suffix)]


def _build_groups(
    model_source: TensorSource,
    base_source: TensorSource,
    *,
    scope: str,
    preset: str,
    include_regexes: list[re.Pattern[str]],
    exclude_regexes: list[re.Pattern[str]],
) -> list[ExtractionGroup]:
    preset_patterns = PRESET_PATTERNS[preset]
    preset_regexes = [] if preset_patterns is None else _compile_regexes(preset_patterns)
    groups: list[ExtractionGroup] = []

    for key in model_source.keys():
        if not key.startswith("model.diffusion_model.") or not key.endswith(".weight"):
            continue
        if scope == "transformer_blocks" and not key.startswith("model.diffusion_model.transformer_blocks."):
            continue
        if not base_source.has(key):
            continue
        shape = model_source.shape(key)
        if len(shape) != 2:
            continue
        if base_source.shape(key) != shape:
            continue
        module = _module_from_key(key)
        if preset_regexes and not _matches_any(module, preset_regexes):
            continue
        identifiers = (key, module)
        if include_regexes and not any(_matches_any(identifier, include_regexes) for identifier in identifiers):
            continue
        if exclude_regexes and any(_matches_any(identifier, exclude_regexes) for identifier in identifiers):
            continue
        groups.append(ExtractionGroup(source_key=key, module=module, shape=(int(shape[0]), int(shape[1]))))

    return sorted(groups, key=lambda group: group.source_key)


def _factorize_exact(
    delta: torch.Tensor,
    max_rank: int,
    sv_epsilon: float,
    svd_driver: str | None,
) -> tuple[torch.Tensor, torch.Tensor, int, float, float, float, float]:
    if delta.is_cuda:
        drivers: list[str | None]
        if svd_driver == "auto":
            drivers = ["gesvda", "gesvdj", None]
        elif svd_driver is not None:
            drivers = [svd_driver]
        else:
            drivers = [None]
        last_exc: Exception | None = None
        for driver in drivers:
            try:
                if driver is None:
                    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
                else:
                    u, s, vh = torch.linalg.svd(delta, full_matrices=False, driver=driver)
                break
            except RuntimeError as exc:
                last_exc = exc
        else:
            raise RuntimeError(f"CUDA SVD failed for shape {tuple(delta.shape)}") from last_exc
    else:
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)

    rank = min(int(max_rank), int(s.numel()))
    if sv_epsilon > 0:
        rank = min(rank, int((s > sv_epsilon).sum().item()))

    total_energy_t = s.square().sum()
    total_energy = float(total_energy_t.item())
    if rank <= 0:
        empty_a = torch.zeros((0, delta.shape[1]), device=delta.device, dtype=torch.float32)
        empty_b = torch.zeros((delta.shape[0], 0), device=delta.device, dtype=torch.float32)
        return empty_a, empty_b, 0, 0.0 if total_energy > 0 else 1.0, 1.0 if total_energy > 0 else 0.0, total_energy, total_energy

    kept = s[:rank]
    sqrt_s = torch.sqrt(kept)
    down = sqrt_s.unsqueeze(1) * vh[:rank, :]
    up = u[:, :rank] * sqrt_s.unsqueeze(0)

    discarded_energy = float(s[rank:].square().sum().item())
    retained_energy = 1.0 - discarded_energy / total_energy if total_energy > 0 else 1.0
    relative_error = math.sqrt(max(discarded_energy, 0.0) / total_energy) if total_energy > 0 else 0.0
    return down, up, rank, retained_energy, relative_error, total_energy, discarded_energy


def _factorize_lowrank(
    delta: torch.Tensor,
    max_rank: int,
    sv_epsilon: float,
    oversample: int,
    niter: int,
) -> tuple[torch.Tensor, torch.Tensor, int, float, float, float, float]:
    target_q = min(int(max_rank) + max(0, int(oversample)), min(delta.shape))
    u, s, v = torch.svd_lowrank(delta, q=target_q, niter=max(0, int(niter)))
    rank = min(int(max_rank), int(s.numel()))
    if sv_epsilon > 0:
        rank = min(rank, int((s > sv_epsilon).sum().item()))

    total_energy = float(delta.square().sum().item())
    if rank <= 0:
        empty_a = torch.zeros((0, delta.shape[1]), device=delta.device, dtype=torch.float32)
        empty_b = torch.zeros((delta.shape[0], 0), device=delta.device, dtype=torch.float32)
        return empty_a, empty_b, 0, 0.0 if total_energy > 0 else 1.0, 1.0 if total_energy > 0 else 0.0, total_energy, total_energy

    kept = s[:rank]
    sqrt_s = torch.sqrt(kept)
    down = sqrt_s.unsqueeze(1) * v[:, :rank].transpose(0, 1)
    up = u[:, :rank] * sqrt_s.unsqueeze(0)

    approx = up @ down
    residual_energy = float((delta - approx).square().sum().item())
    retained_energy = 1.0 - residual_energy / total_energy if total_energy > 0 else 1.0
    relative_error = math.sqrt(max(residual_energy, 0.0) / total_energy) if total_energy > 0 else 0.0
    del approx
    return down, up, rank, retained_energy, relative_error, total_energy, residual_energy


def _factorize_delta(
    delta: torch.Tensor,
    *,
    method: str,
    max_rank: int,
    sv_epsilon: float,
    svd_driver: str | None,
    lowrank_oversample: int,
    lowrank_niter: int,
) -> tuple[torch.Tensor, torch.Tensor, int, float, float, float, float]:
    if method == "exact":
        return _factorize_exact(delta, max_rank, sv_epsilon, svd_driver)
    return _factorize_lowrank(delta, max_rank, sv_epsilon, lowrank_oversample, lowrank_niter)


def _output_keys(module: str, fmt: str) -> tuple[str, str, str | None]:
    if fmt == "comfy":
        base = f"diffusion_model.{module}"
        return f"{base}.lora_A.weight", f"{base}.lora_B.weight", None
    base = f"lora_unet_model_{module.replace('.', '_')}"
    return f"{base}.lora_down.weight", f"{base}.lora_up.weight", f"{base}.alpha"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Target/fine-tuned LTX-2 checkpoint")
    parser.add_argument("--base-model", required=True, type=Path, help="Base/source LTX-2 checkpoint")
    parser.add_argument("--out-path", type=Path, help="Output LoRA safetensors path")
    parser.add_argument("--report-path", type=Path, help="Optional extraction report JSON path")
    parser.add_argument("--format", choices=("comfy", "musubi"), default="comfy", help="Output LoRA key format")
    parser.add_argument(
        "--scope",
        choices=("transformer_blocks", "diffusion_model"),
        default="transformer_blocks",
        help="Tensor scope. Both options exclude VAE/vocoder/text components.",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESET_PATTERNS.keys()),
        default="video_attn_ffn",
        help="Musubi-style LoRA target preset to extract",
    )
    parser.add_argument("--max-rank", type=int, default=32, help="Maximum LoRA rank per module")
    parser.add_argument("--sv-epsilon", type=float, default=0.0, help="Drop singular values <= epsilon")
    parser.add_argument("--min-delta-norm", type=float, default=0.0, help="Skip tensors with Frobenius norm <= this value")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Device for per-module SVD")
    parser.add_argument("--svd-method", choices=("exact", "lowrank"), default="lowrank", help="SVD implementation")
    parser.add_argument("--lowrank-oversample", type=int, default=8, help="Extra q columns for torch.svd_lowrank")
    parser.add_argument("--lowrank-niter", type=int, default=2, help="Power iterations for torch.svd_lowrank")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for lowrank SVD")
    parser.add_argument(
        "--svd-driver",
        choices=("auto", "gesvda", "gesvdj", "default"),
        default="auto",
        help="CUDA exact-SVD driver preference",
    )
    parser.add_argument("--save-dtype", choices=tuple(DTYPE_BY_NAME.keys()), default="bfloat16")
    parser.add_argument("--include-regex", action="append", default=[], help="Only include matching source/module names")
    parser.add_argument("--exclude-regex", action="append", default=[], help="Exclude matching source/module names")
    parser.add_argument("--max-groups", type=int, help="Process only the first N matched groups")
    parser.add_argument("--list-groups", action="store_true", help="List matched groups and exit")
    parser.add_argument("--torch-threads", type=int, help="Optional torch CPU thread cap")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_rank <= 0:
        raise ValueError("--max-rank must be > 0")
    if args.svd_method == "lowrank" and args.lowrank_oversample < 0:
        raise ValueError("--lowrank-oversample must be >= 0")

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available")
    device = torch.device(device_name)

    if args.torch_threads is not None:
        torch.set_num_threads(max(1, int(args.torch_threads)))
        try:
            torch.set_num_interop_threads(max(1, int(args.torch_threads)))
        except RuntimeError:
            pass

    torch.manual_seed(int(args.seed))
    save_dtype = DTYPE_BY_NAME[args.save_dtype]
    svd_driver = "auto" if args.svd_driver == "auto" else None if args.svd_driver == "default" else args.svd_driver

    base_source = TensorSource(args.base_model)
    model_source = TensorSource(args.model)
    groups = _build_groups(
        model_source,
        base_source,
        scope=args.scope,
        preset=args.preset,
        include_regexes=_compile_regexes(args.include_regex),
        exclude_regexes=_compile_regexes(args.exclude_regex),
    )
    if args.max_groups is not None:
        groups = groups[: max(0, int(args.max_groups))]

    print("[LTX2Extractor] Starting extraction")
    print(f"[LTX2Extractor] Base model: {args.base_model}")
    print(f"[LTX2Extractor] Model:      {args.model}")
    print(f"[LTX2Extractor] Format:     {args.format}")
    print(f"[LTX2Extractor] Scope:      {args.scope}")
    print(f"[LTX2Extractor] Preset:     {args.preset}")
    print(f"[LTX2Extractor] Max rank:   {args.max_rank}")
    print(f"[LTX2Extractor] SVD:        {args.svd_method} on {device.type}")
    print(f"[LTX2Extractor] Save dtype: {args.save_dtype}")
    print(f"[LTX2Extractor] Groups:     {len(groups)}")

    if args.list_groups:
        for group in groups:
            print(f"{group.module}\t{group.shape[0]}x{group.shape[1]}")
        return

    if args.out_path is None:
        raise RuntimeError("--out-path is required unless --list-groups is used")
    if not groups:
        raise RuntimeError("No extraction groups matched the requested filters")

    output: dict[str, torch.Tensor] = {}
    report_items: list[ExtractionReportItem] = []
    started_at = time.time()

    for group in tqdm(groups, desc="Extracting LTX2 LoRA groups", unit="group"):
        model_tensor = model_source.get_tensor(group.source_key).to(device=device, dtype=torch.float32, non_blocking=True)
        base_tensor = base_source.get_tensor(group.source_key).to(device=device, dtype=torch.float32, non_blocking=True)
        delta = model_tensor - base_tensor
        del model_tensor, base_tensor

        delta_norm = float(torch.linalg.norm(delta).item())
        if delta_norm <= float(args.min_delta_norm):
            report_items.append(
                ExtractionReportItem(
                    source_key=group.source_key,
                    module=group.module,
                    tensor_shape=group.shape,
                    status="skipped_zero_delta",
                    actual_rank=0,
                    max_rank=int(args.max_rank),
                    retained_energy=1.0,
                    relative_frobenius_error=0.0,
                    total_energy=0.0,
                    discarded_energy=0.0,
                    svd_method=args.svd_method,
                )
            )
            del delta
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        down, up, rank, retained, rel_err, total_energy, discarded_energy = _factorize_delta(
            delta,
            method=args.svd_method,
            max_rank=int(args.max_rank),
            sv_epsilon=float(args.sv_epsilon),
            svd_driver=svd_driver,
            lowrank_oversample=int(args.lowrank_oversample),
            lowrank_niter=int(args.lowrank_niter),
        )
        report_items.append(
            ExtractionReportItem(
                source_key=group.source_key,
                module=group.module,
                tensor_shape=group.shape,
                status="extracted" if rank > 0 else "skipped_rank_zero",
                actual_rank=int(rank),
                max_rank=int(args.max_rank),
                retained_energy=float(retained),
                relative_frobenius_error=float(rel_err),
                total_energy=float(total_energy),
                discarded_energy=float(discarded_energy),
                svd_method=args.svd_method,
            )
        )

        if rank > 0:
            down_key, up_key, alpha_key = _output_keys(group.module, args.format)
            output[down_key] = down.to(device="cpu", dtype=save_dtype).contiguous()
            output[up_key] = up.to(device="cpu", dtype=save_dtype).contiguous()
            if alpha_key is not None:
                output[alpha_key] = torch.tensor(float(rank), dtype=save_dtype)

        del delta, down, up
        if device.type == "cuda":
            torch.cuda.empty_cache()

    extracted = [item for item in report_items if item.status == "extracted"]
    if not output:
        raise RuntimeError("No LoRA tensors were extracted; all groups were skipped")

    total_energy = sum(item.total_energy for item in extracted)
    discarded = sum(item.discarded_energy for item in extracted)
    global_retained = 1.0 - discarded / total_energy if total_energy > 0 else 1.0
    global_rel_err = math.sqrt(max(discarded, 0.0) / total_energy) if total_energy > 0 else 0.0
    mean_retained = sum(item.retained_energy for item in extracted) / len(extracted)
    mean_rel_err = sum(item.relative_frobenius_error for item in extracted) / len(extracted)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "ss_output_name": args.out_path.stem,
        "ss_network_module": "networks.lora_ltx2",
        "ss_network_dim": str(args.max_rank),
        "ss_network_alpha": str(args.max_rank),
        "ss_ltx2_diff_lora": "true",
        "ss_ltx2_diff_base_model": str(args.base_model),
        "ss_ltx2_diff_target_model": str(args.model),
        "ss_ltx2_diff_scope": args.scope,
        "ss_ltx2_diff_preset": args.preset,
        "ss_ltx2_diff_svd_method": args.svd_method,
    }
    print(f"[LTX2Extractor] Writing output: {args.out_path}")
    save_file(output, str(args.out_path), metadata=metadata)

    print("[LTX2Extractor] Extraction quality")
    print(f"[LTX2Extractor] Extracted groups:             {len(extracted)}")
    print(f"[LTX2Extractor] Output tensors:               {len(output)}")
    print(f"[LTX2Extractor] Mean retained energy:        {mean_retained:.6f}")
    print(f"[LTX2Extractor] Mean relative frob error:   {mean_rel_err:.6f}")
    print(f"[LTX2Extractor] Global retained energy:     {global_retained:.6f}")
    print(f"[LTX2Extractor] Global relative frob err:  {global_rel_err:.6f}")
    print(f"[LTX2Extractor] Elapsed seconds:            {time.time() - started_at:.1f}")
    print("[LTX2Extractor] Lowest retained-energy groups:")
    for item in sorted(extracted, key=lambda entry: entry.retained_energy)[:5]:
        print(
            f"  - {item.module}: retained={item.retained_energy:.6f} "
            f"rel_err={item.relative_frobenius_error:.6f} rank={item.actual_rank}"
        )

    report_path = args.report_path
    if report_path is None:
        report_path = args.out_path.with_suffix(args.out_path.suffix + ".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "base_model": str(args.base_model),
                    "model": str(args.model),
                    "out_path": str(args.out_path),
                    "format": args.format,
                    "scope": args.scope,
                    "preset": args.preset,
                    "max_rank": int(args.max_rank),
                    "sv_epsilon": float(args.sv_epsilon),
                    "svd_method": args.svd_method,
                    "device": device.type,
                    "save_dtype": args.save_dtype,
                    "groups": len(report_items),
                    "extracted_groups": len(extracted),
                    "mean_retained_energy": mean_retained,
                    "mean_relative_frobenius_error": mean_rel_err,
                    "global_retained_energy": global_retained,
                    "global_relative_frobenius_error": global_rel_err,
                    "elapsed_seconds": time.time() - started_at,
                },
                "groups": [asdict(item) for item in report_items],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[LTX2Extractor] Wrote report: {report_path}")


if __name__ == "__main__":
    main()
