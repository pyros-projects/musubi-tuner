#!/usr/bin/env python3
"""Probe Krea2 prompt sensitivity to a direct txtfusion projector bypass patch."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


BYPASS_DIFF_KEYS = (
    "diffusion_model.txtfusion.projector.diff",
    "txtfusion.projector.diff",
    "diffusion_model.txtfusion.projector.weight.diff",
    "txtfusion.projector.weight.diff",
)


@dataclass(frozen=True)
class PromptRecord:
    source: str
    prompt: str


@dataclass(frozen=True)
class SensitivityMetrics:
    fusion_delta: float
    cosine_delta: float
    score: float


@dataclass(frozen=True)
class ReportRow:
    source: str
    prompt: str
    tokens: int
    fusion_delta: float
    cosine_delta: float
    score: float
    verdict: str


def _non_empty_record(source: str, text: str) -> PromptRecord | None:
    prompt = text.strip()
    if not prompt:
        return None
    return PromptRecord(source=source, prompt=prompt)


def collect_prompt_records(
    *,
    prompts: Iterable[str] | None,
    prompt_file: str | Path | None,
    sidecar_dir: str | Path | None,
    recursive: bool = False,
) -> list[PromptRecord]:
    records: list[PromptRecord] = []

    for index, prompt in enumerate(prompts or [], start=1):
        record = _non_empty_record(f"cli:{index}", prompt)
        if record is not None:
            records.append(record)

    if prompt_file is not None:
        path = Path(prompt_file)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            record = _non_empty_record(f"{path}:{line_number}", line)
            if record is not None:
                records.append(record)

    if sidecar_dir is not None:
        root = Path(sidecar_dir)
        pattern = "**/*.txt" if recursive else "*.txt"
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            record = _non_empty_record(str(path), path.read_text(encoding="utf-8"))
            if record is not None:
                records.append(record)

    return records


def sort_report_rows(rows: list[ReportRow], sort: str) -> list[ReportRow]:
    if sort == "input":
        return list(rows)
    if sort == "score":
        return sorted(rows, key=lambda row: row.score, reverse=True)
    raise ValueError(f"Unsupported sort mode: {sort}")


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(0, width - 3)] + "..."


def format_table(rows: list[ReportRow]) -> str:
    columns = [
        ("score", lambda row: _format_float(row.score)),
        ("verdict", lambda row: row.verdict),
        ("fusion_delta", lambda row: _format_float(row.fusion_delta)),
        ("cosine_delta", lambda row: _format_float(row.cosine_delta)),
        ("tokens", lambda row: str(row.tokens)),
        ("source", lambda row: row.source),
        ("prompt", lambda row: row.prompt),
    ]
    values = [[getter(row) for _, getter in columns] for row in rows]
    widths = []
    for index, (name, _) in enumerate(columns):
        max_value = max([len(name), *(len(row[index]) for row in values)], default=len(name))
        cap = 88 if name == "prompt" else 56 if name == "source" else max_value
        widths.append(min(max_value, cap))

    header = "  ".join(name.ljust(widths[index]) for index, (name, _) in enumerate(columns))
    divider = "  ".join("-" * width for width in widths)
    lines = [header, divider]
    for row_values in values:
        lines.append("  ".join(_truncate(value, widths[index]).ljust(widths[index]) for index, value in enumerate(row_values)))
    return "\n".join(lines)


def export_report(rows: list[ReportRow], output_path: str | Path, export_format: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(row) for row in rows]
    if export_format == "csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()) if data else [field.name for field in ReportRow.__dataclass_fields__.values()])
            writer.writeheader()
            writer.writerows(data)
    elif export_format == "json":
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


def infer_export_format(output_path: str | Path | None, explicit_format: str) -> str | None:
    if output_path is None:
        return None
    if explicit_format != "auto":
        return explicit_format
    suffix = Path(output_path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    raise ValueError("Could not infer export format from output extension; pass --format csv or --format json")


def load_bypass_projector_diff(path: str | Path) -> torch.Tensor:
    import torch
    from safetensors.torch import load_file

    state_dict = load_file(str(path), device="cpu")
    for key in BYPASS_DIFF_KEYS:
        if key in state_dict:
            diff = state_dict[key].detach().to(torch.float32)
            if diff.shape != (1, 12):
                raise ValueError(f"Bypass projector diff has unsupported shape {tuple(diff.shape)}; expected (1, 12)")
            return diff
    raise ValueError(f"No supported txtfusion projector diff found in {path}")


def apply_projector_diff(projector: torch.nn.Linear, diff: torch.Tensor, *, strength: float) -> None:
    import torch

    if tuple(projector.weight.shape) != tuple(diff.shape):
        raise ValueError(f"Projector shape {tuple(projector.weight.shape)} does not match diff shape {tuple(diff.shape)}")
    with torch.no_grad():
        projector.weight.add_(diff.to(device=projector.weight.device, dtype=projector.weight.dtype) * strength)


def compute_sensitivity_metrics(baseline: torch.Tensor, patched: torch.Tensor) -> SensitivityMetrics:
    import torch
    import torch.nn.functional as F

    base = baseline.detach().float()
    changed = patched.detach().float()
    diff = changed - base
    denom = max(float(base.abs().mean().item()), 1.0)
    fusion_delta = float(diff.abs().mean().item() / denom)

    base_flat = base.reshape(1, -1)
    changed_flat = changed.reshape(1, -1)
    base_norm = float(base_flat.norm().item())
    changed_norm = float(changed_flat.norm().item())
    if base_norm < 1e-12 or changed_norm < 1e-12:
        cosine_delta = 0.0 if torch.allclose(base_flat, changed_flat) else 1.0
    else:
        cosine_delta = float((1.0 - F.cosine_similarity(base_flat, changed_flat).item()))
    score = cosine_delta * 100.0
    return SensitivityMetrics(fusion_delta=fusion_delta, cosine_delta=cosine_delta, score=score)


def verdict_for_score(score: float) -> str:
    if score < 4.0:
        return "stable"
    if score < 8.0:
        return "mild-redirect"
    if score < 18.0:
        return "redirected"
    return "strongly-redirected"


def str_to_dtype(name: str) -> torch.dtype:
    import torch

    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def resolve_device(device: str) -> torch.device:
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def clean_device_memory(device: torch.device) -> None:
    import torch

    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def build_textfusion(device: torch.device, dtype: torch.dtype):
    from musubi_tuner.krea2.krea2_mmdit import TextFusionTransformer
    from musubi_tuner.krea2.krea2_utils import single_mmdit_large_wide

    config = single_mmdit_large_wide
    textfusion = TextFusionTransformer(
        config.txtlayers,
        config.txtdim,
        config.txtheads,
        config.multiplier,
        config.bias,
        config.txtkvheads,
    )
    return textfusion.to(device=device, dtype=dtype).eval().requires_grad_(False)


def _textfusion_key_for_checkpoint_key(key: str) -> str | None:
    prefixes = (
        "txtfusion.",
        "diffusion_model.txtfusion.",
        "model.diffusion_model.txtfusion.",
    )
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return None


def load_textfusion_weights(textfusion: torch.nn.Module, dit_path: str | Path, device: torch.device, dtype: torch.dtype) -> None:
    from safetensors import safe_open

    expected = set(textfusion.state_dict().keys())
    state_dict = {}
    with safe_open(str(dit_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            module_key = _textfusion_key_for_checkpoint_key(key)
            if module_key is None:
                continue
            if module_key in expected:
                state_dict[module_key] = f.get_tensor(key).to(device=device, dtype=dtype)

    missing = sorted(expected - set(state_dict.keys()))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Krea2 text-fusion weights could not be loaded from {dit_path}; missing: {preview}")
    textfusion.load_state_dict(state_dict, strict=True)


def encode_prompt_batch(records: list[PromptRecord], text_encoder_path: str | Path, device: torch.device, dtype: torch.dtype):
    import torch

    from musubi_tuner.krea2 import krea2_utils
    from musubi_tuner.krea2.krea2_sampling import gather_valid_text

    encoder = None
    with torch.no_grad():
        encoder = krea2_utils.load_krea2_text_encoder(str(text_encoder_path), dtype=dtype, device=device)
        hiddens, mask = krea2_utils.get_krea2_prompt_embeds(encoder, [record.prompt for record in records])
        hiddens, mask = gather_valid_text(hiddens, mask)
    del encoder
    clean_device_memory(device)
    return hiddens.to(device=device, dtype=dtype), mask.to(device=device)


def run_textfusion(textfusion: torch.nn.Module, text: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
    import torch

    from musubi_tuner.modules.attention import AttentionParams

    nomask = AttentionParams.create_attention_params_from_mask("torch", False, 0, None)
    masked = AttentionParams.create_attention_params_from_mask("torch", False, 0, text_mask)
    with torch.no_grad():
        return textfusion(text, nomask, masked)


def score_encoded_prompts(
    records: list[PromptRecord],
    textfusion: torch.nn.Module,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    bypass_diff: torch.Tensor,
    bypass_strength: float,
) -> list[ReportRow]:
    import torch

    original_projector = textfusion.projector.weight.detach().clone()
    with torch.no_grad():
        baseline = run_textfusion(textfusion, text, text_mask)
        try:
            apply_projector_diff(textfusion.projector, bypass_diff, strength=bypass_strength)
            patched = run_textfusion(textfusion, text, text_mask)
        finally:
            textfusion.projector.weight.copy_(original_projector)

    rows: list[ReportRow] = []
    for index, record in enumerate(records):
        valid_len = int(text_mask[index].sum().item())
        metrics = compute_sensitivity_metrics(baseline[index : index + 1, :valid_len], patched[index : index + 1, :valid_len])
        rows.append(
            ReportRow(
                source=record.source,
                prompt=record.prompt,
                tokens=valid_len,
                fusion_delta=metrics.fusion_delta,
                cosine_delta=metrics.cosine_delta,
                score=metrics.score,
                verdict=verdict_for_score(metrics.score),
            )
        )
    return rows


def score_prompts(args: argparse.Namespace, records: list[PromptRecord]) -> list[ReportRow]:
    device = resolve_device(args.device)
    dtype = str_to_dtype(args.dtype)
    textfusion = build_textfusion(device, dtype)
    load_textfusion_weights(textfusion, args.dit, device, dtype)
    bypass_diff = load_bypass_projector_diff(args.bypass)

    rows: list[ReportRow] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        text = text_mask = None
        try:
            text, text_mask = encode_prompt_batch(batch, args.text_encoder, device, dtype)
            rows.extend(score_encoded_prompts(batch, textfusion, text, text_mask, bypass_diff, args.bypass_strength))
        finally:
            del text, text_mask
            clean_device_memory(device)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Krea2 prompt sensitivity by comparing txtfusion conditioning before and after a direct projector bypass patch. "
            "This is a bypass-sensitivity diagnostic, not a definitive safety or acceptance classifier."
        )
    )
    parser.add_argument("--prompt", action="append", default=None, help="Prompt to score. Can be repeated.")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Text file with one prompt per line.")
    parser.add_argument("--sidecar-dir", type=Path, default=None, help="Folder containing .txt image sidecar captions.")
    parser.add_argument("--recursive", action="store_true", help="Read sidecar .txt files recursively.")
    parser.add_argument("--output", type=Path, default=None, help="Optional report export path (.csv or .json).")
    parser.add_argument("--format", choices=("auto", "csv", "json"), default="auto", help="Export format. Default infers from --output.")
    parser.add_argument("--sort", choices=("score", "input"), default="score", help="Report ordering.")
    parser.add_argument("--batch-size", type=int, default=1, help="Prompt batch size for text encoding/scoring.")
    parser.add_argument("--device", default="auto", help="Torch device for real scoring: auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--dtype", default="bf16", help="Real scoring dtype: bf16, fp16, or fp32.")
    parser.add_argument("--dit", type=Path, default=None, help="Krea2 DiT safetensors path. Only txtfusion weights are loaded.")
    parser.add_argument("--text_encoder", type=Path, default=None, help="Qwen3-VL text encoder safetensors path.")
    parser.add_argument("--bypass", type=Path, default=None, help="Direct Krea2 txtfusion projector bypass safetensors.")
    parser.add_argument(
        "--bypass-strength",
        "--bypass-weight",
        dest="bypass_strength",
        type=float,
        default=1.0,
        help="Multiplier for the bypass projector diff. Use values like 5.0 to match stronger ComfyUI bypass weights.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.prompt and args.prompt_file is None and args.sidecar_dir is None:
        parser.error("Provide at least one input: --prompt, --prompt-file, or --sidecar-dir")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    for label in ("dit", "text_encoder", "bypass"):
        if getattr(args, label) is None:
            parser.error(f"--{label} is required for scoring")
    if args.prompt_file is not None and not args.prompt_file.is_file():
        parser.error(f"--prompt-file is not a file: {args.prompt_file}")
    if args.sidecar_dir is not None and not args.sidecar_dir.is_dir():
        parser.error(f"--sidecar-dir is not a directory: {args.sidecar_dir}")
    for label in ("dit", "text_encoder", "bypass"):
        path = getattr(args, label)
        if path is not None and not path.is_file():
            parser.error(f"--{label} is not a file: {path}")
    try:
        infer_export_format(args.output, args.format)
        str_to_dtype(args.dtype)
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    records = collect_prompt_records(
        prompts=args.prompt,
        prompt_file=args.prompt_file,
        sidecar_dir=args.sidecar_dir,
        recursive=args.recursive,
    )
    if not records:
        parser.error("No non-empty prompts were found")

    rows = sort_report_rows(score_prompts(args, records), args.sort)
    print(format_table(rows))

    export_format = infer_export_format(args.output, args.format)
    if args.output is not None and export_format is not None:
        export_report(rows, args.output, export_format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
