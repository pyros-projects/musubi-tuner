import sys
from argparse import Namespace
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musubi_tuner.krea2_train_network import Krea2NetworkTrainer


class _FakeKrea2Attention(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.wq = torch.nn.Linear(4, 4, bias=False)
        self.wq.weight.data.zero_()
        if dtype != torch.float32:
            self.wq.weight = torch.nn.Parameter(torch.zeros((4, 4), dtype=dtype), requires_grad=False)


class _FakeKrea2Block(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.attn = _FakeKrea2Attention(dtype=dtype)


class _FakeKrea2Transformer(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_FakeKrea2Block(dtype=dtype)])


def _native_wq_lora_state():
    return {
        "diffusion_model.blocks.0.attn.wq.lora_down.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.wq.lora_up.weight": torch.ones(4, 2),
        "diffusion_model.blocks.0.attn.wq.alpha": torch.tensor(2.0),
    }


def _native_wq_lora_state_without_alpha():
    return {
        "diffusion_model.blocks.0.attn.wq.lora_down.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.wq.lora_up.weight": torch.ones(4, 2),
    }


def test_native_krea2_turbo_lora_keys_are_normalized_for_sampling_lora():
    trainer = Krea2NetworkTrainer()
    weights_sd = {
        "diffusion_model.blocks.0.attn.wq.lora_down.weight": torch.randn(2, 4),
        "diffusion_model.blocks.0.attn.wq.lora_up.weight": torch.randn(4, 2),
        "diffusion_model.blocks.0.attn.wq.alpha": torch.tensor(2.0),
        "diffusion_model.txtfusion.refiner_blocks.1.mlp.down.lora_down.weight": torch.randn(2, 4),
        "diffusion_model.txtfusion.refiner_blocks.1.mlp.down.lora_up.weight": torch.randn(4, 2),
    }

    normalized = trainer.normalize_sampling_lora_weights(Namespace(), weights_sd, "krea2_turbo.safetensors")

    assert "lora_unet_blocks_0_attn_wq.lora_down.weight" in normalized
    assert "lora_unet_blocks_0_attn_wq.lora_up.weight" in normalized
    assert "lora_unet_blocks_0_attn_wq.alpha" in normalized
    assert "lora_unet_txtfusion_refiner_blocks_1_mlp_down.lora_down.weight" in normalized
    assert "lora_unet_txtfusion_refiner_blocks_1_mlp_down.lora_up.weight" in normalized
    assert not any(key.startswith("diffusion_model.") for key in normalized)


def test_native_krea2_turbo_lora_without_alpha_defaults_alpha_to_rank():
    trainer = Krea2NetworkTrainer()
    transformer = _FakeKrea2Transformer()
    args = Namespace(network_module=None)
    normalized = trainer.normalize_sampling_lora_weights(args, _native_wq_lora_state_without_alpha(), "krea2_turbo.safetensors")

    assert normalized["lora_unet_blocks_0_attn_wq.alpha"].item() == 2.0

    with mock.patch.object(trainer, "_load_sampling_lora_weights", return_value=normalized):
        applied = trainer._apply_sampling_lora_specs(
            args,
            transformer,
            torch.device("cpu"),
            [("krea2_turbo.safetensors", 0.5)],
        )

    assert applied


def test_sampling_lora_apply_and_restore_uses_krea2_default_module():
    trainer = Krea2NetworkTrainer()
    transformer = _FakeKrea2Transformer()
    args = Namespace(network_module=None)
    normalized = trainer.normalize_sampling_lora_weights(args, _native_wq_lora_state(), "krea2_turbo.safetensors")

    with mock.patch.object(trainer, "_load_sampling_lora_weights", return_value=normalized):
        applied = trainer._apply_sampling_lora_specs(
            args,
            transformer,
            torch.device("cpu"),
            [("krea2_turbo.safetensors", 0.5)],
        )

    assert applied
    assert torch.allclose(transformer.blocks[0].attn.wq.weight, torch.ones_like(transformer.blocks[0].attn.wq.weight))

    trainer._restore_sampling_lora_specs(args, transformer, torch.device("cpu"), applied)

    assert torch.count_nonzero(transformer.blocks[0].attn.wq.weight) == 0


def test_sampling_lora_float8_module_uses_runtime_overlay_and_clears_on_restore():
    trainer = Krea2NetworkTrainer()
    transformer = _FakeKrea2Transformer(dtype=torch.float8_e4m3fn)
    args = Namespace(network_module=None)
    normalized = trainer.normalize_sampling_lora_weights(args, _native_wq_lora_state(), "krea2_turbo.safetensors")

    with mock.patch.object(trainer, "_load_sampling_lora_weights", return_value=normalized):
        applied = trainer._apply_sampling_lora_specs(
            args,
            transformer,
            torch.device("cpu"),
            [("krea2_turbo.safetensors", 1.0)],
        )

    module = transformer.blocks[0].attn.wq
    assert applied
    assert hasattr(module, "_sampling_lora_runtime_adapters")
    assert not applied[0][1]

    trainer._restore_sampling_lora_specs(args, transformer, torch.device("cpu"), applied)

    assert not hasattr(module, "_sampling_lora_runtime_adapters")
    assert not hasattr(module, "_sampling_lora_runtime_forward")
