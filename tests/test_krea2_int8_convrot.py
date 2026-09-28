import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musubi_tuner.krea2 import krea2_sampling, krea2_utils
from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer
from musubi_tuner.modules.int8_optimization_utils import _convrot_hadamard, _quantize_int8_rowwise
from musubi_tuner.hv_train_network import NetworkTrainer
from musubi_tuner.networks import lora_krea2
from musubi_tuner.utils.train_lora_overlay import set_train_lora_overlay_enabled

TINY = SingleMMDiTConfig(
    features=128, tdim=32, txtdim=32, heads=2, kvheads=1, multiplier=2, layers=2, patch=2, channels=4,
    txtlayers=3, txtheads=2, txtkvheads=2,
)  # fmt: skip


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    model = SingleStreamDiT(TINY)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.1)
    return model.eval()


def _write_int8_checkpoint(model, path, group_size=64):
    """Comfy layout: the main blocks' Linears as INT8 ConvRot, everything else as is."""
    sd = {}
    for key, value in model.state_dict().items():
        module = key[: -len(".weight")]
        if key.startswith("blocks.") and key.endswith(".weight") and value.ndim == 2:
            q, scale = _quantize_int8_rowwise(_convrot_hadamard(value.float(), group_size))
            sd[key], sd[f"{module}.weight_scale"] = q, scale
            marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size}
            sd[f"{module}.comfy_quant"] = torch.tensor(list(json.dumps(marker).encode()), dtype=torch.uint8)
        else:
            sd[key] = value.contiguous()
    save_file(sd, str(path))


def _inputs(batch=2, txt_lens=(5, 3), seed=1):
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(batch, TINY.channels, 8, 8, generator=g)
    txt_len = max(txt_lens)
    context = torch.randn(batch, txt_len, TINY.txtlayers, TINY.txtdim, generator=g)
    txtmask = torch.zeros(batch, txt_len, dtype=torch.bool)
    for i, n in enumerate(txt_lens):
        txtmask[i, :n] = True
    img, pos, mask = krea2_sampling.prepare(noise, txt_len, TINY.patch, txtmask)
    return dict(img=img, context=context, t=torch.rand(batch, generator=g), pos=pos, mask=mask)


def test_int8_convrot_krea2_dit_loads_frozen_int8_and_trains(tmp_path):
    reference = _tiny_model()
    path = tmp_path / "krea2_tiny_int8_convrot.safetensors"
    _write_int8_checkpoint(reference, path)

    assert len(krea2_utils.krea2_int8_convrot_layers(str(path))) == 8 * TINY.layers
    model = krea2_utils.load_krea2_dit(str(path), device="cpu", config=TINY)
    block = model.blocks[0]
    for linear in (block.attn.wq, block.attn.wk, block.attn.gate, block.mlp.gate, block.mlp.down):
        assert linear.weight.dtype == torch.int8 and not linear.weight.requires_grad
    assert model.first.weight.dtype == torch.float32  # non-quantized tensors keep their checkpoint dtype

    inputs = _inputs()
    with torch.no_grad():
        expected = reference(**inputs)
    img = inputs["img"].clone().requires_grad_(True)
    out = model(**{**inputs, "img": img})
    assert ((out - expected).norm() / expected.norm()).item() < 0.05  # INT8 weight + activation noise
    out.square().mean().backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()


def test_int8_convrot_krea2_rejects_fp8_and_lora_merging(tmp_path):
    path = tmp_path / "krea2_tiny_int8_convrot.safetensors"
    _write_int8_checkpoint(_tiny_model(), path)
    with pytest.raises(ValueError, match="fp8"):
        krea2_utils.load_krea2_dit(str(path), device="cpu", config=TINY, fp8_scaled=True)
    with pytest.raises(ValueError, match="LoRA"):
        krea2_utils.load_krea2_dit(str(path), device="cpu", config=TINY, lora_weights=[{}])


def test_sampling_lora_overlays_int8_layers_instead_of_merging(tmp_path):
    path = tmp_path / "krea2_tiny_int8_convrot.safetensors"
    _write_int8_checkpoint(_tiny_model(), path)
    model = krea2_utils.load_krea2_dit(str(path), device="cpu", config=TINY)
    wq, first = model.blocks[0].attn.wq, model.first
    wq_before, first_before = wq.weight.detach().clone(), first.weight.detach().clone()

    g = torch.Generator().manual_seed(2)
    weights = {}
    for name, linear in (("blocks_0_attn_wq", wq), ("first", first)):
        weights[f"lora_unet_{name}.lora_down.weight"] = torch.randn(4, linear.in_features, generator=g)
        weights[f"lora_unet_{name}.lora_up.weight"] = torch.randn(linear.out_features, 4, generator=g)
        weights[f"lora_unet_{name}.alpha"] = torch.tensor(4.0)
    network = lora_krea2.create_arch_network_from_weights(1.0, weights, unet=model, for_inference=True)

    trainer = Krea2NetworkTrainer()
    merged, attached, backups = trainer._apply_sampling_lora_network(network, weights, torch.device("cpu"))
    assert (merged, attached) == (1, 1)  # the INT8 wq gets an overlay, the fp32 first layer is merged
    torch.testing.assert_close(wq.weight, wq_before)
    assert not torch.equal(first.weight, first_before)

    trainer._clear_sampling_lora_runtime_overlays(model)
    trainer._restore_sampling_lora_network(backups)
    torch.testing.assert_close(first.weight, first_before)


def _write_ai_toolkit_lora(reference, path, names, rank=2, seed=7):
    """ai-toolkit / PEFT layout, like the TextFusion refusal-reduction LoRA: diffusion_model. prefix, no alpha."""
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for name in names:
        linear = reference.get_submodule(name)
        sd[f"diffusion_model.{name}.lora_A.weight"] = torch.randn(rank, linear.in_features, generator=g) * 0.2
        sd[f"diffusion_model.{name}.lora_B.weight"] = torch.randn(linear.out_features, rank, generator=g) * 0.2
    save_file(sd, str(path))


def test_train_lora_overlay_is_frozen_unmerged_on_int8_and_never_saved(tmp_path, monkeypatch):
    reference = _tiny_model()
    ckpt = tmp_path / "krea2_tiny_int8_convrot.safetensors"
    _write_int8_checkpoint(reference, ckpt)
    overlay_path = tmp_path / "refusal.safetensors"
    names = ("txtfusion.layerwise_blocks.0.attn.wq", "txtfusion.refiner_blocks.1.attn.wo", "blocks.0.attn.wq")
    _write_ai_toolkit_lora(reference, overlay_path, names)

    int8_model = krea2_utils.load_krea2_dit(str(ckpt), device="cpu", config=TINY)
    monkeypatch.setattr(krea2_utils, "load_krea2_dit", lambda *a, **k: int8_model)
    inputs = _inputs()
    with torch.no_grad():
        base = int8_model(**inputs)

    trainer = Krea2NetworkTrainer()
    trainer.blocks_to_swap = 0
    args = argparse.Namespace(
        fp8_base=False, fp8_scaled=False, compile=False, bypass=None, network_module=None,
        train_lora_overlay=f"{overlay_path}:0.5",
    )  # fmt: skip
    trainer.handle_model_specific_args(args)
    model = trainer.load_transformer(SimpleNamespace(device=torch.device("cpu")), args, str(ckpt), "torch", False, "cpu", torch.float32)
    overlay = trainer.train_lora_overlay
    assert len(overlay.unet_loras) == len(names)
    assert all(lora.multiplier == 0.5 for lora in overlay.unet_loras)
    assert not any(p.requires_grad for p in overlay.parameters())
    assert model.blocks[0].attn.wq.weight.dtype == torch.int8  # unmerged: the INT8 payload is untouched
    with torch.no_grad():
        assert not torch.allclose(model(**inputs), base)

    # the trained LoRA wraps the overlaid modules: gradients reach it, never the overlay
    network = lora_krea2.create_arch_network(1.0, 4, 4, None, [], model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    model(**inputs).square().mean().backward()
    assert all(p.grad is None for p in overlay.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters())

    # like --bypass, sample images keep the overlay on: no sample_images override
    assert Krea2NetworkTrainer.sample_images is NetworkTrainer.sample_images
    assert trainer.get_checkpoint_metadata(args) == {
        "ss_krea2_train_lora_overlay": "refusal.safetensors",
        "ss_krea2_train_lora_overlay_strength": "0.5",
    }

    set_train_lora_overlay_enabled(overlay, False)  # the trained LoRA's up weights are still zero-initialized
    with torch.no_grad():
        torch.testing.assert_close(model(**inputs), base)
