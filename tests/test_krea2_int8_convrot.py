import json
import sys
from pathlib import Path

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
from musubi_tuner.networks import lora_krea2

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
