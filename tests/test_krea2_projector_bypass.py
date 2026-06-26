import argparse
import importlib
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musubi_tuner import krea2_generate_image
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser


def _projector_bypass_module():
    try:
        return importlib.import_module("musubi_tuner.krea2.projector_bypass")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Krea2 projector bypass helper is missing: {exc}")


class _FakeTextFusion(torch.nn.Module):
    def __init__(self, shape=(1, 12), dtype=torch.bfloat16):
        super().__init__()
        self.projector = torch.nn.Linear(shape[1], shape[0], bias=False, dtype=dtype)
        self.projector.weight.data.zero_()


class _FakeKrea2DiT(torch.nn.Module):
    def __init__(self, shape=(1, 12), dtype=torch.bfloat16):
        super().__init__()
        self.txtfusion = _FakeTextFusion(shape=shape, dtype=dtype)
        self.events: list[str] = []

    def eval(self):
        self.events.append("eval")
        return self

    def requires_grad_(self, requires_grad=False):
        self.events.append(f"requires_grad:{requires_grad}")
        return self

    def to(self, device):
        self.events.append(f"to:{torch.device(device).type}")
        return self

    def enable_block_swap(self, *args, **kwargs):
        self.events.append("enable_block_swap")

    def move_to_device_except_swap_blocks(self, device):
        self.events.append(f"move_except:{torch.device(device).type}")

    def switch_block_swap_for_inference(self):
        self.events.append("switch_block_swap_for_inference")


class _FakeVAE:
    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def requires_grad_(self, requires_grad=False):
        return self


def test_supported_projector_bypass_diff_keys_are_loaded(tmp_path):
    projector_bypass = _projector_bypass_module()
    diff = torch.arange(12, dtype=torch.float32).reshape(1, 12)

    for key in projector_bypass.SUPPORTED_PROJECTOR_DIFF_KEYS:
        path = tmp_path / f"{key.replace('.', '_')}.safetensors"
        save_file({key: diff}, path)

        loaded = projector_bypass.load_projector_bypass_diff(path)

        assert torch.equal(loaded, diff)


def test_missing_projector_bypass_diff_key_fails_clearly(tmp_path):
    projector_bypass = _projector_bypass_module()
    path = tmp_path / "missing.safetensors"
    save_file({"diffusion_model.blocks.0.attn.wq.lora_down.weight": torch.ones(1, 1)}, path)

    with pytest.raises(ValueError, match="projector diff"):
        projector_bypass.load_projector_bypass_diff(path)


def test_projector_bypass_applies_weighted_diff_and_preserves_projector_dtype():
    projector_bypass = _projector_bypass_module()
    model = _FakeKrea2DiT(dtype=torch.bfloat16)
    diff = torch.arange(12, dtype=torch.float32).reshape(1, 12)

    projector_bypass.apply_projector_bypass(model, diff, weight=5)

    assert model.txtfusion.projector.weight.dtype == torch.bfloat16
    assert torch.allclose(model.txtfusion.projector.weight.float(), diff * 5)


def test_projector_bypass_shape_mismatch_fails_clearly():
    projector_bypass = _projector_bypass_module()
    model = _FakeKrea2DiT(shape=(1, 12))

    with pytest.raises(ValueError, match="shape"):
        projector_bypass.apply_projector_bypass(model, torch.ones(1, 11), weight=1)


def test_training_parser_accepts_bypass_args():
    parser = krea2_setup_parser(argparse.ArgumentParser())

    args = parser.parse_args(["--bypass", "/tmp/bypass.safetensors", "--bypass-weight", "5"])
    alias_args = parser.parse_args(["--bypass", "/tmp/bypass.safetensors", "--bypass_weight", "6"])

    assert args.bypass == "/tmp/bypass.safetensors"
    assert args.bypass_weight == 5
    assert alias_args.bypass_weight == 6


def test_generate_parser_accepts_bypass_args():
    with mock.patch.object(
        sys,
        "argv",
        [
            "krea2_generate_image.py",
            "hello",
            "--dit",
            "/tmp/dit.safetensors",
            "--vae",
            "/tmp/vae.safetensors",
            "--text_encoder",
            "/tmp/text_encoder.safetensors",
            "--save_path",
            "/tmp/out",
            "--bypass",
            "/tmp/bypass.safetensors",
            "--bypass-weight",
            "5",
        ],
    ):
        args = krea2_generate_image.parse_args()

    assert args.bypass == "/tmp/bypass.safetensors"
    assert args.bypass_weight == 5


def test_generate_parser_accepts_bypass_weight_alias():
    with mock.patch.object(
        sys,
        "argv",
        [
            "krea2_generate_image.py",
            "hello",
            "--dit",
            "/tmp/dit.safetensors",
            "--vae",
            "/tmp/vae.safetensors",
            "--text_encoder",
            "/tmp/text_encoder.safetensors",
            "--save_path",
            "/tmp/out",
            "--bypass",
            "/tmp/bypass.safetensors",
            "--bypass_weight",
            "6",
        ],
    ):
        args = krea2_generate_image.parse_args()

    assert args.bypass_weight == 6


def test_training_metadata_records_active_bypass_only():
    trainer = Krea2NetworkTrainer()

    active = trainer.get_checkpoint_metadata(types.SimpleNamespace(bypass="/tmp/bypass.safetensors", bypass_weight=5))
    inactive = trainer.get_checkpoint_metadata(types.SimpleNamespace(bypass=None, bypass_weight=1))

    assert active["ss_krea2_bypass_path"] == "/tmp/bypass.safetensors"
    assert active["ss_krea2_bypass_weight"] == "5"
    assert "ss_krea2_bypass_path" not in inactive
    assert "ss_krea2_bypass_weight" not in inactive


def test_training_load_transformer_applies_bypass_after_dit_load(tmp_path):
    diff_path = tmp_path / "bypass.safetensors"
    diff = torch.ones(1, 12)
    save_file({"diffusion_model.txtfusion.projector.diff": diff}, diff_path)
    model = _FakeKrea2DiT(dtype=torch.float32)
    trainer = Krea2NetworkTrainer()
    args = types.SimpleNamespace(
        fp8_scaled=False,
        bypass=str(diff_path),
        bypass_weight=5,
    )

    with mock.patch("musubi_tuner.krea2_train_network.krea2_utils.load_krea2_dit", return_value=model) as load_mock:
        result = trainer.load_transformer(
            mock.Mock(),
            args,
            "/tmp/dit.safetensors",
            "torch",
            False,
            "cpu",
            torch.bfloat16,
        )

    assert result is model
    assert load_mock.called
    assert torch.allclose(model.txtfusion.projector.weight, diff * 5)


def test_build_pipeline_applies_bypass_before_eval_and_block_swap(tmp_path):
    diff_path = tmp_path / "bypass.safetensors"
    diff = torch.ones(1, 12)
    save_file({"diffusion_model.txtfusion.projector.diff": diff}, diff_path)
    model = _FakeKrea2DiT(dtype=torch.float32)

    with (
        mock.patch("musubi_tuner.krea2_generate_image.qwen_image_utils.load_vae", return_value=_FakeVAE()),
        mock.patch("musubi_tuner.krea2_generate_image.krea2_utils.load_krea2_dit", return_value=model),
    ):
        dit, _vae = krea2_generate_image.build_pipeline(
            "/tmp/dit.safetensors",
            "/tmp/vae.safetensors",
            device="cpu",
            dtype=torch.bfloat16,
            bypass=str(diff_path),
            bypass_weight=5,
        )

    assert dit is model
    assert torch.allclose(model.txtfusion.projector.weight, diff * 5)
    assert model.events[:2] == ["eval", "requires_grad:False"]
