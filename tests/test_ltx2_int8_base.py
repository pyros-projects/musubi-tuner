"""Tests for the INT8 ConvRot base lane of the LTX-2 trainer (checkpoint scan + patch glue)."""

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from musubi_tuner.ltx2_train_network import _scan_ltx2_int8_convrot_layers  # noqa: E402
from musubi_tuner.modules.int8_optimization_utils import (  # noqa: E402
    _convrot_hadamard,
    apply_int8_convrot_monkey_patch,
)

GROUP = 4


def _marker(fmt="int8_tensorwise", convrot=True, group=GROUP) -> torch.Tensor:
    payload = json.dumps({"format": fmt, "convrot": convrot, "convrot_groupsize": group})
    return torch.frombuffer(bytearray(payload.encode()), dtype=torch.uint8).clone()


def _quantized_linear(out_f: int, in_f: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    w_rot = torch.randn(out_f, in_f, generator=g)
    scale = (w_rot.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-30)
    return {
        "weight": (w_rot / scale).round().clamp(-128, 127).to(torch.int8),
        "weight_scale": scale.float(),
    }


class TestScan:
    def test_detects_renames_and_parses(self, tmp_path):
        q = _quantized_linear(8, 8, 1)
        tensors = {
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": q["weight"],
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale": q["weight_scale"],
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.comfy_quant": _marker(),
            "model.diffusion_model.patchify_proj.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
        }
        path = tmp_path / "int8.safetensors"
        save_file(tensors, str(path))
        layers = _scan_ltx2_int8_convrot_layers([str(path)])
        assert list(layers) == ["transformer_blocks.0.attn1.to_q"]
        config = layers["transformer_blocks.0.attn1.to_q"]
        assert config.group_size == GROUP
        assert config.scale_shape == (8, 1)

    def test_weight_level_marker_layout(self, tmp_path):
        q = _quantized_linear(8, 8, 2)
        tensors = {
            "model.diffusion_model.transformer_blocks.1.ff.net.2.weight": q["weight"],
            "model.diffusion_model.transformer_blocks.1.ff.net.2.weight_scale": q["weight_scale"],
            "model.diffusion_model.transformer_blocks.1.ff.net.2.weight.comfy_quant": _marker(),
        }
        path = tmp_path / "int8b.safetensors"
        save_file(tensors, str(path))
        layers = _scan_ltx2_int8_convrot_layers([str(path)])
        assert list(layers) == ["transformer_blocks.1.ff.net.2"]

    def test_bf16_checkpoint_returns_empty(self, tmp_path):
        path = tmp_path / "bf16.safetensors"
        save_file({"model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": torch.zeros(4, 4, dtype=torch.bfloat16)}, str(path))
        assert _scan_ltx2_int8_convrot_layers([str(path)]) == {}

    def test_nvfp4_rejected(self, tmp_path):
        q = _quantized_linear(8, 8, 3)
        tensors = {
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": q["weight"],
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale": q["weight_scale"],
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.comfy_quant": _marker(fmt="nvfp4"),
        }
        path = tmp_path / "nvfp4.safetensors"
        save_file(tensors, str(path))
        with pytest.raises(ValueError, match="int8 ConvRot"):
            _scan_ltx2_int8_convrot_layers([str(path)])

    def test_missing_scale_rejected(self, tmp_path):
        q = _quantized_linear(8, 8, 4)
        tensors = {
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": q["weight"],
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.comfy_quant": _marker(),
        }
        path = tmp_path / "noscale.safetensors"
        save_file(tensors, str(path))
        with pytest.raises(ValueError, match="weight_scale"):
            _scan_ltx2_int8_convrot_layers([str(path)])


class TestPatchLoadForward:
    def test_scan_patch_load_forward_matches_dense_reference(self, tmp_path):
        """The full glue: scan configs -> patch meta model -> load_state_dict(assign) -> forward."""
        q = _quantized_linear(8, 8, 5)
        tensors = {
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": q["weight"],
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale": q["weight_scale"],
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.comfy_quant": _marker(),
        }
        path = tmp_path / "glue.safetensors"
        save_file(tensors, str(path))
        layers = _scan_ltx2_int8_convrot_layers([str(path)])

        class Attn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(8, 8, bias=False)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = Attn()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = torch.nn.ModuleList([Block()])

        with torch.device("meta"):
            model = Model()
        apply_int8_convrot_monkey_patch(model, layers)
        sd = {
            "transformer_blocks.0.attn1.to_q.weight": q["weight"],
            "transformer_blocks.0.attn1.to_q.weight_scale": q["weight_scale"],
        }
        model.load_state_dict(sd, strict=False, assign=True)

        module = model.transformer_blocks[0].attn1.to_q
        assert module.weight.dtype == torch.int8
        assert not module.weight.requires_grad

        x = torch.randn(2, 8, requires_grad=True)
        out = module(x)
        w_dense = _convrot_hadamard(q["weight"].float() * q["weight_scale"], GROUP)
        ref = torch.nn.functional.linear(x, w_dense)
        torch.testing.assert_close(out, ref, rtol=0.05, atol=0.05)
        # Input gradients flow (needed for LoRA below the patched layer).
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
