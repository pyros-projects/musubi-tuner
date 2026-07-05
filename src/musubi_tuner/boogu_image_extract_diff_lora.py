#!/usr/bin/env python3
"""Extract a Boogu Image diff LoRA from two transformer checkpoints.

The output matches the local Boogu ComfyUI diff-LoRA convention:

- 2D ``*.weight`` tensors become ``diffusion_model.<module>.lora_down/up.weight``.
- Non-2D ``*.weight`` tensors become ``diffusion_model.<module>.diff``.
- ``*.bias`` tensors become ``diffusion_model.<module>.diff_b``.

This is useful for comparing related full Boogu transformer checkpoints, such as
``boogu_image_edit_turbo_bf16 - boogu_image_edit_bf16``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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


@dataclass(frozen=True)
class LoraGroup:
    source_key: str
    module: str
    shape: tuple[int, int]
    down_key: str
    up_key: str


@dataclass(frozen=True)
class DirectGroup:
    source_key: str
    shape: tuple[int, ...]
    output_key: str


@dataclass(frozen=True)
class ExtractionGroups:
    lora: list[LoraGroup]
    direct: list[DirectGroup]
    skipped: list[dict[str, object]]


@dataclass(frozen=True)
class ExtractionReportItem:
    source_key: str
    status: str
    kind: str
    shape: tuple[int, ...]
    delta_norm: float
    output_key: str | None = None
    down_key: str | None = None
    up_key: str | None = None
    module: str | None = None
    actual_rank: int = 0
    max_rank: int = 0
    retained_energy: float = 1.0
    relative_frobenius_error: float = 0.0
    total_energy: float = 0.0
    discarded_energy: float = 0.0


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
            return handle.get_tensor(key)


def _compile_regexes(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def _matches_any(text: str, regexes: list[re.Pattern[str]]) -> bool:
    return any(regex.fullmatch(text) or regex.search(text) for regex in regexes)


def _module_from_weight_key(key: str) -> str:
    return key[: -len(".weight")]


def _module_from_bias_key(key: str) -> str:
    return key[: -len(".bias")]


def _lora_output_keys(module: str) -> tuple[str, str]:
    base = f"diffusion_model.{module}"
    return f"{base}.lora_down.weight", f"{base}.lora_up.weight"


def _direct_output_key(key: str) -> str:
    if key.endswith(".bias"):
        return f"diffusion_model.{_module_from_bias_key(key)}.diff_b"
    if key.endswith(".weight"):
        return f"diffusion_model.{_module_from_weight_key(key)}.diff"
    raise ValueError(f"Unsupported direct Boogu tensor key: {key}")


def _keep_key(key: str, include_regexes: list[re.Pattern[str]], exclude_regexes: list[re.Pattern[str]]) -> bool:
    if include_regexes and not _matches_any(key, include_regexes):
        return False
    return not (exclude_regexes and _matches_any(key, exclude_regexes))


def build_extraction_groups(
    model_source: TensorSource,
    base_source: TensorSource,
    *,
    include_regexes: list[re.Pattern[str]] | None = None,
    exclude_regexes: list[re.Pattern[str]] | None = None,
    max_groups: int | None = None,
) -> ExtractionGroups:
    include_regexes = include_regexes or []
    exclude_regexes = exclude_regexes or []
    lora: list[LoraGroup] = []
    direct: list[DirectGroup] = []
    skipped: list[dict[str, object]] = []

    for key in model_source.keys():
        if not base_source.has(key):
            continue
        if not _keep_key(key, include_regexes, exclude_regexes):
            continue

        shape = model_source.shape(key)
        if base_source.shape(key) != shape:
            skipped.append(
                {
                    "key": key,
                    "status": "skipped_shape_mismatch",
                    "target_shape": list(shape),
                    "base_shape": list(base_source.shape(key)),
                }
            )
            continue

        if key.endswith(".weight") and len(shape) == 2:
            module = _module_from_weight_key(key)
            down_key, up_key = _lora_output_keys(module)
            lora.append(LoraGroup(key, module, (int(shape[0]), int(shape[1])), down_key, up_key))
        elif key.endswith(".weight") or key.endswith(".bias"):
            direct.append(DirectGroup(key, tuple(int(dim) for dim in shape), _direct_output_key(key)))
        else:
            skipped.append({"key": key, "status": "skipped_unsupported_key", "shape": list(shape)})

    lora = sorted(lora, key=lambda group: group.source_key)
    direct = sorted(direct, key=lambda group: group.source_key)
    skipped = sorted(skipped, key=lambda item: str(item["key"]))
    if max_groups is not None:
        lora = lora[: max(0, int(max_groups))]
    return ExtractionGroups(lora=lora, direct=direct, skipped=skipped)


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
        empty_down = torch.zeros((0, delta.shape[1]), device=delta.device, dtype=torch.float32)
        empty_up = torch.zeros((delta.shape[0], 0), device=delta.device, dtype=torch.float32)
        retained = 0.0 if total_energy > 0 else 1.0
        relative_error = 1.0 if total_energy > 0 else 0.0
        return empty_down, empty_up, 0, retained, relative_error, total_energy, total_energy

    kept = s[:rank]
    sqrt_s = torch.sqrt(kept)
    down = sqrt_s.unsqueeze(1) * vh[:rank, :]
    up = u[:, :rank] * sqrt_s.unsqueeze(0)

    discarded_energy = float(s[rank:].square().sum().item())
    retained_energy = 1.0 - discarded_energy / total_energy if total_energy > 0 else 1.0
    relative_error = math.sqrt(max(discarded_energy, 0.0) / total_energy) if total_energy > 0 else 0.0
    return down, up, rank, retained_energy, relative_error, total_energy, discarded_energy


def extract_diff_lora(
    *,
    model_path: Path,
    base_model_path: Path,
    out_path: Path,
    report_path: Path | None = None,
    max_rank: int = 128,
    sv_epsilon: float = 0.0,
    min_delta_norm: float = 0.0,
    device: torch.device | None = None,
    save_dtype: torch.dtype = torch.bfloat16,
    svd_driver: str | None = "auto",
    include_regexes: list[re.Pattern[str]] | None = None,
    exclude_regexes: list[re.Pattern[str]] | None = None,
    max_groups: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if max_rank <= 0:
        raise ValueError("max_rank must be > 0")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists, use --overwrite to replace it: {out_path}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA extraction but CUDA is not available")

    base_source = TensorSource(base_model_path)
    model_source = TensorSource(model_path)
    groups = build_extraction_groups(
        model_source,
        base_source,
        include_regexes=include_regexes,
        exclude_regexes=exclude_regexes,
        max_groups=max_groups,
    )

    output: dict[str, torch.Tensor] = {}
    report_items: list[ExtractionReportItem] = []
    started_at = time.time()

    for group in tqdm(groups.direct, desc="Direct Boogu diffs", unit="tensor"):
        model_tensor = model_source.get_tensor(group.source_key).to(torch.float32)
        base_tensor = base_source.get_tensor(group.source_key).to(torch.float32)
        delta = model_tensor - base_tensor
        delta_norm = float(torch.linalg.norm(delta).item())
        if delta_norm <= float(min_delta_norm):
            report_items.append(
                ExtractionReportItem(
                    source_key=group.source_key,
                    shape=group.shape,
                    status="skipped_zero_delta",
                    kind="direct",
                    delta_norm=delta_norm,
                )
            )
            continue
        output[group.output_key] = delta.to(dtype=save_dtype, device="cpu").contiguous()
        report_items.append(
            ExtractionReportItem(
                source_key=group.source_key,
                shape=group.shape,
                status="diff",
                kind="direct",
                delta_norm=delta_norm,
                output_key=group.output_key,
            )
        )

    for group in tqdm(groups.lora, desc="SVD Boogu diff LoRA", unit="group"):
        model_tensor = model_source.get_tensor(group.source_key).to(device=device, dtype=torch.float32, non_blocking=True)
        base_tensor = base_source.get_tensor(group.source_key).to(device=device, dtype=torch.float32, non_blocking=True)
        delta = model_tensor - base_tensor
        del model_tensor, base_tensor

        delta_norm = float(torch.linalg.norm(delta).item())
        if delta_norm <= float(min_delta_norm):
            report_items.append(
                ExtractionReportItem(
                    source_key=group.source_key,
                    shape=group.shape,
                    status="skipped_zero_delta",
                    kind="lora",
                    delta_norm=delta_norm,
                    module=group.module,
                    actual_rank=0,
                    max_rank=int(max_rank),
                )
            )
            del delta
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        down, up, rank, retained, rel_err, total_energy, discarded_energy = _factorize_exact(
            delta,
            max_rank=int(max_rank),
            sv_epsilon=float(sv_epsilon),
            svd_driver=svd_driver,
        )
        if rank > 0:
            output[group.down_key] = down.to(device="cpu", dtype=save_dtype).contiguous()
            output[group.up_key] = up.to(device="cpu", dtype=save_dtype).contiguous()
        report_items.append(
            ExtractionReportItem(
                source_key=group.source_key,
                shape=group.shape,
                status="extracted" if rank > 0 else "skipped_rank_zero",
                kind="lora",
                delta_norm=delta_norm,
                module=group.module,
                down_key=group.down_key,
                up_key=group.up_key,
                actual_rank=int(rank),
                max_rank=int(max_rank),
                retained_energy=float(retained),
                relative_frobenius_error=float(rel_err),
                total_energy=float(total_energy),
                discarded_energy=float(discarded_energy),
            )
        )
        del delta, down, up
        if device.type == "cuda":
            torch.cuda.empty_cache()

    extracted_lora = [item for item in report_items if item.status == "extracted"]
    direct_diffs = [item for item in report_items if item.status == "diff"]
    if not output:
        raise RuntimeError("No diff LoRA tensors were extracted; all groups were skipped")

    total_energy = sum(item.total_energy for item in extracted_lora)
    discarded = sum(item.discarded_energy for item in extracted_lora)
    global_retained = 1.0 - discarded / total_energy if total_energy > 0 else 1.0
    global_rel_err = math.sqrt(max(discarded, 0.0) / total_energy) if total_energy > 0 else 0.0
    mean_retained = sum(item.retained_energy for item in extracted_lora) / len(extracted_lora) if extracted_lora else 1.0
    mean_rel_err = (
        sum(item.relative_frobenius_error for item in extracted_lora) / len(extracted_lora) if extracted_lora else 0.0
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    metadata = {
        "ss_output_name": out_path.stem,
        "ss_network_module": "networks.lora_boogu_image",
        "ss_network_dim": str(max_rank),
        "ss_network_alpha": str(max_rank),
        "ss_boogu_diff_lora": "true",
        "ss_boogu_diff_base_model": str(base_model_path),
        "ss_boogu_diff_target_model": str(model_path),
        "ss_boogu_diff_svd_method": "exact",
        "ss_boogu_diff_save_dtype": str(save_dtype).replace("torch.", ""),
    }
    save_file(output, str(tmp_path), metadata=metadata)
    os.replace(tmp_path, out_path)

    elapsed = time.time() - started_at
    summary: dict[str, object] = {
        "base_model": str(base_model_path),
        "target_model": str(model_path),
        "out_path": str(out_path),
        "format": "boogu_comfy_diff",
        "max_rank": int(max_rank),
        "sv_epsilon": float(sv_epsilon),
        "svd_method": "exact",
        "svd_driver": svd_driver or "default",
        "device": device.type,
        "save_dtype": str(save_dtype).replace("torch.", ""),
        "lora_groups": len(groups.lora),
        "extracted_lora_groups": len(extracted_lora),
        "direct_groups": len(groups.direct),
        "direct_diff_groups": len(direct_diffs),
        "skipped_groups": len(groups.skipped),
        "output_tensors": len(output),
        "mean_retained_energy": mean_retained,
        "mean_relative_frobenius_error": mean_rel_err,
        "global_retained_energy": global_retained,
        "global_relative_frobenius_error": global_rel_err,
        "elapsed_seconds": elapsed,
    }

    if report_path is None:
        report_path = out_path.with_suffix(out_path.suffix + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "skipped": groups.skipped,
                "groups": [asdict(item) for item in report_items],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Target Boogu transformer checkpoint")
    parser.add_argument("--base-model", required=True, type=Path, help="Base/source Boogu transformer checkpoint")
    parser.add_argument("--out-path", type=Path, help="Output diff LoRA safetensors path")
    parser.add_argument("--report-path", type=Path, help="Optional extraction report JSON path")
    parser.add_argument("--max-rank", type=int, default=128, help="Maximum LoRA rank per 2D weight")
    parser.add_argument("--sv-epsilon", type=float, default=0.0, help="Drop singular values <= epsilon")
    parser.add_argument("--min-delta-norm", type=float, default=0.0, help="Skip tensors with Frobenius norm <= this value")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Device for per-module SVD")
    parser.add_argument(
        "--svd-driver",
        choices=("auto", "gesvda", "gesvdj", "default"),
        default="auto",
        help="CUDA SVD driver preference",
    )
    parser.add_argument("--save-dtype", choices=tuple(DTYPE_BY_NAME.keys()), default="bfloat16")
    parser.add_argument("--include-regex", action="append", default=[], help="Only include matching source keys")
    parser.add_argument("--exclude-regex", action="append", default=[], help="Exclude matching source keys")
    parser.add_argument("--max-groups", type=int, help="Process only the first N 2D LoRA groups")
    parser.add_argument("--list-groups", action="store_true", help="List matched groups and exit")
    parser.add_argument("--torch-threads", type=int, help="Optional torch CPU thread cap")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(max(1, int(args.torch_threads)))
        try:
            torch.set_num_interop_threads(max(1, int(args.torch_threads)))
        except RuntimeError:
            pass

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    svd_driver = "auto" if args.svd_driver == "auto" else None if args.svd_driver == "default" else args.svd_driver
    include_regexes = _compile_regexes(args.include_regex)
    exclude_regexes = _compile_regexes(args.exclude_regex)

    base_source = TensorSource(args.base_model)
    model_source = TensorSource(args.model)
    groups = build_extraction_groups(
        model_source,
        base_source,
        include_regexes=include_regexes,
        exclude_regexes=exclude_regexes,
        max_groups=args.max_groups,
    )
    print("[BooguDiff] Starting extraction")
    print(f"[BooguDiff] Base model: {args.base_model}")
    print(f"[BooguDiff] Model:      {args.model}")
    print(f"[BooguDiff] Max rank:   {args.max_rank}")
    print(f"[BooguDiff] SVD:        exact on {device.type}")
    print(f"[BooguDiff] Save dtype: {args.save_dtype}")
    print(f"[BooguDiff] LoRA groups:   {len(groups.lora)}")
    print(f"[BooguDiff] Direct groups: {len(groups.direct)}")
    print(f"[BooguDiff] Skipped keys:   {len(groups.skipped)}")

    if args.list_groups:
        for group in groups.direct:
            print(f"direct\t{group.source_key}\t{group.shape}\t{group.output_key}")
        for group in groups.lora:
            print(f"lora\t{group.source_key}\t{group.shape}\t{group.down_key}\t{group.up_key}")
        for item in groups.skipped:
            print(f"skipped\t{item['key']}\t{item['status']}")
        return
    if args.out_path is None:
        raise RuntimeError("--out-path is required unless --list-groups is used")

    summary = extract_diff_lora(
        model_path=args.model,
        base_model_path=args.base_model,
        out_path=args.out_path,
        report_path=args.report_path,
        max_rank=int(args.max_rank),
        sv_epsilon=float(args.sv_epsilon),
        min_delta_norm=float(args.min_delta_norm),
        device=device,
        save_dtype=DTYPE_BY_NAME[args.save_dtype],
        svd_driver=svd_driver,
        include_regexes=include_regexes,
        exclude_regexes=exclude_regexes,
        max_groups=args.max_groups,
        overwrite=bool(args.overwrite),
    )
    print(f"[BooguDiff] Wrote output: {summary['out_path']}")
    print(f"[BooguDiff] Output tensors: {summary['output_tensors']}")
    print(f"[BooguDiff] Extracted LoRA groups: {summary['extracted_lora_groups']}")
    print(f"[BooguDiff] Direct diff groups: {summary['direct_diff_groups']}")
    print(f"[BooguDiff] Mean retained energy: {summary['mean_retained_energy']:.6f}")
    print(f"[BooguDiff] Global retained energy: {summary['global_retained_energy']:.6f}")
    print(f"[BooguDiff] Global relative frob err: {summary['global_relative_frobenius_error']:.6f}")
    print(f"[BooguDiff] Elapsed seconds: {summary['elapsed_seconds']:.1f}")


if __name__ == "__main__":
    main()
