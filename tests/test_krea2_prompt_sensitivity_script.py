import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "krea2_prompt_sensitivity.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("krea2_prompt_sensitivity", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collects_cli_prompts_with_provenance():
    mod = load_script_module()

    prompts = mod.collect_prompt_records(prompts=["alpha", " beta "], prompt_file=None, sidecar_dir=None)

    assert [(p.source, p.prompt) for p in prompts] == [("cli:1", "alpha"), ("cli:2", "beta")]


def test_collects_non_empty_prompt_file_lines_with_line_provenance(tmp_path):
    mod = load_script_module()
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("first\n\n  second  \n", encoding="utf-8")

    prompts = mod.collect_prompt_records(prompts=None, prompt_file=prompt_file, sidecar_dir=None)

    assert [(p.source, p.prompt) for p in prompts] == [(f"{prompt_file}:1", "first"), (f"{prompt_file}:3", "second")]


def test_collects_sidecars_deterministically_and_recursively_when_requested(tmp_path):
    mod = load_script_module()
    (tmp_path / "b.txt").write_text("bee", encoding="utf-8")
    (tmp_path / "a.txt").write_text("aye", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.txt").write_text("see", encoding="utf-8")

    flat = mod.collect_prompt_records(prompts=None, prompt_file=None, sidecar_dir=tmp_path, recursive=False)
    recursive = mod.collect_prompt_records(prompts=None, prompt_file=None, sidecar_dir=tmp_path, recursive=True)

    assert [(p.source, p.prompt) for p in flat] == [(str(tmp_path / "a.txt"), "aye"), (str(tmp_path / "b.txt"), "bee")]
    assert [(p.source, p.prompt) for p in recursive] == [
        (str(tmp_path / "a.txt"), "aye"),
        (str(tmp_path / "b.txt"), "bee"),
        (str(nested / "c.txt"), "see"),
    ]


def test_blank_prompts_are_ignored(tmp_path):
    mod = load_script_module()
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("\n   \n", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("  ", encoding="utf-8")

    prompts = mod.collect_prompt_records(prompts=[" ", "real"], prompt_file=prompt_file, sidecar_dir=tmp_path)

    assert [(p.source, p.prompt) for p in prompts] == [("cli:2", "real")]


def test_report_rows_sort_render_and_export(tmp_path):
    mod = load_script_module()
    rows = [
        mod.ReportRow("cli:1", "low", 3, 0.01, 0.02, 0.02, "stable"),
        mod.ReportRow("cli:2", "high", 4, 0.40, 0.20, 0.40, "bypass-sensitive"),
    ]

    sorted_rows = mod.sort_report_rows(rows, sort="score")
    table = mod.format_table(sorted_rows)
    csv_path = tmp_path / "report.csv"
    json_path = tmp_path / "report.json"
    mod.export_report(sorted_rows, csv_path, "csv")
    mod.export_report(sorted_rows, json_path, "json")

    assert [row.prompt for row in sorted_rows] == ["high", "low"]
    assert "bypass-sensitive" in table
    assert "fusion_delta" in table

    with csv_path.open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[0]["prompt"] == "high"

    json_rows = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_rows[0]["source"] == "cli:2"


def test_export_format_inference_and_validation():
    mod = load_script_module()

    assert mod.infer_export_format("report.csv", "auto") == "csv"
    assert mod.infer_export_format("report.json", "auto") == "json"
    assert mod.infer_export_format("report.txt", "csv") == "csv"
    assert mod.infer_export_format(None, "auto") is None
    with pytest.raises(ValueError, match="infer export format"):
        mod.infer_export_format("report.txt", "auto")


def test_bypass_weight_alias_sets_bypass_strength():
    mod = load_script_module()

    args = mod.build_parser().parse_args(
        [
            "--prompt",
            "test prompt",
            "--dit",
            "/tmp/dit.safetensors",
            "--text_encoder",
            "/tmp/text_encoder.safetensors",
            "--bypass",
            "/tmp/bypass.safetensors",
            "--bypass-weight",
            "5",
        ]
    )

    assert args.bypass_strength == 5.0


def test_bypass_projector_diff_is_loaded_and_applied(tmp_path):
    mod = load_script_module()
    bypass = tmp_path / "bypass.safetensors"
    diff = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    save_file({"diffusion_model.txtfusion.projector.diff": diff}, bypass)
    projector = torch.nn.Linear(12, 1, bias=False)
    projector.weight.data.zero_()

    loaded = mod.load_bypass_projector_diff(bypass)
    mod.apply_projector_diff(projector, loaded, strength=0.5)

    assert torch.equal(loaded, diff)
    assert torch.equal(projector.weight, diff * 0.5)


def test_missing_bypass_projector_diff_fails_clearly(tmp_path):
    mod = load_script_module()
    bypass = tmp_path / "bad.safetensors"
    save_file({"some.other.key": torch.ones(1)}, bypass)

    with pytest.raises(ValueError, match="projector diff"):
        mod.load_bypass_projector_diff(bypass)


def test_score_metrics_and_verdicts_use_absolute_cosine_redirect():
    mod = load_script_module()
    base = torch.zeros(1, 2, 3)
    patched = torch.ones(1, 2, 3) * 0.2

    metrics = mod.compute_sensitivity_metrics(base, patched)

    assert metrics.fusion_delta == pytest.approx(0.2)
    assert metrics.cosine_delta == pytest.approx(1.0)
    assert metrics.score == pytest.approx(100.0)
    assert mod.verdict_for_score(3.0) == "stable"
    assert mod.verdict_for_score(5.0) == "mild-redirect"
    assert mod.verdict_for_score(10.0) == "redirected"
    assert mod.verdict_for_score(20.0) == "strongly-redirected"


def test_redirect_score_uses_absolute_cosine_delta_scale():
    mod = load_script_module()
    base = torch.tensor([[[1.0, 0.0]]])
    patched = torch.tensor([[[0.0, 1.0]]])

    metrics = mod.compute_sensitivity_metrics(base, patched)

    assert metrics.cosine_delta == pytest.approx(1.0)
    assert metrics.score == pytest.approx(100.0)


def test_score_prompts_releases_batch_resources_between_encodes(monkeypatch):
    mod = load_script_module()
    records = [
        mod.PromptRecord("cli:1", "first"),
        mod.PromptRecord("cli:2", "second"),
    ]
    events = []

    class FakeTensor:
        def __init__(self, name):
            self.name = name

    def fake_encode(batch, *_args):
        events.append(f"encode:{batch[0].prompt}")
        return FakeTensor(f"text:{batch[0].prompt}"), FakeTensor(f"mask:{batch[0].prompt}")

    def fake_score(batch, _textfusion, text, _text_mask, _bypass_diff, _bypass_strength):
        events.append(f"score:{batch[0].prompt}:{text.name}")
        return [mod.ReportRow(batch[0].source, batch[0].prompt, 1, 0.0, 0.04, 4.0, "mild-redirect")]

    def fake_clean(device):
        events.append(f"clean:{device.type}")

    monkeypatch.setattr(mod, "build_textfusion", lambda *_args: object())
    monkeypatch.setattr(mod, "load_textfusion_weights", lambda *_args: None)
    monkeypatch.setattr(mod, "load_bypass_projector_diff", lambda *_args: object())
    monkeypatch.setattr(mod, "encode_prompt_batch", fake_encode)
    monkeypatch.setattr(mod, "score_encoded_prompts", fake_score)
    monkeypatch.setattr(mod, "clean_device_memory", fake_clean, raising=False)

    args = type(
        "Args",
        (),
        {
            "device": "cpu",
            "dtype": "fp32",
            "dit": "dit.safetensors",
            "text_encoder": "text_encoder.safetensors",
            "bypass": "bypass.safetensors",
            "bypass_strength": 1.0,
            "batch_size": 1,
        },
    )()

    rows = mod.score_prompts(args, records)

    assert [row.prompt for row in rows] == ["first", "second"]
    assert [row.score for row in rows] == [4.0, 4.0]
    assert events == [
        "encode:first",
        "score:first:text:first",
        "clean:cpu",
        "encode:second",
        "score:second:text:second",
        "clean:cpu",
    ]


def test_partial_textfusion_weight_loader_accepts_native_and_diffusion_model_keys(tmp_path):
    mod = load_script_module()

    class TinyTextFusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projector = torch.nn.Linear(12, 1, bias=False)

    native = tmp_path / "native.safetensors"
    prefixed = tmp_path / "prefixed.safetensors"
    weight = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    save_file({"txtfusion.projector.weight": weight}, native)
    save_file({"diffusion_model.txtfusion.projector.weight": weight + 1}, prefixed)

    native_module = TinyTextFusion()
    prefixed_module = TinyTextFusion()
    mod.load_textfusion_weights(native_module, native, torch.device("cpu"), torch.float32)
    mod.load_textfusion_weights(prefixed_module, prefixed, torch.device("cpu"), torch.float32)

    assert torch.equal(native_module.projector.weight, weight)
    assert torch.equal(prefixed_module.projector.weight, weight + 1)


def test_partial_textfusion_weight_loader_reports_missing_keys(tmp_path):
    mod = load_script_module()

    class TinyTextFusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projector = torch.nn.Linear(12, 1, bias=False)

    path = tmp_path / "empty.safetensors"
    save_file({"blocks.0.weight": torch.ones(1)}, path)

    with pytest.raises(ValueError, match="text-fusion weights"):
        mod.load_textfusion_weights(TinyTextFusion(), path, torch.device("cpu"), torch.float32)
