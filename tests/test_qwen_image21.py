import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musubi_tuner.modules.int8_optimization_utils import _convrot_hadamard, _quantize_int8_rowwise
from musubi_tuner.networks import lora_qwen_image21
from musubi_tuner.qwen_image21 import qwen_image21_utils
from musubi_tuner.qwen_image21.qwen_image21_model import QwenImage21Transformer2DModel
from musubi_tuner.qwen_image21.qwen_image21_text_encoder import Int8ConvRotEmbedding, _te_key
from musubi_tuner.qwen_image21.qwen_image21_vae import QwenImage21VAE

TINY = dict(
    in_channels=8,
    out_channels=8,
    num_layers=2,
    attention_head_dim=16,
    num_attention_heads=4,
    context_in_dim=32,
    mlp_ratio=3,
    axes_dims_rope=(4, 6, 6),
)

# the launcher's LORA_CONFIG presets (.pyro/qwen21/train.sh)
PRESET_1 = r"transformer_blocks\.[0-9]+\.attn\.(to_q|to_k|to_v|to_out\.0)"
PRESET_2 = r"transformer_blocks\.[0-9]+\.(attn\.(to_q|to_k|to_v|to_out\.0)|img_mlp\.(gate_up|out))"
PRESET_3 = r".*blocks\..*"


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    model = QwenImage21Transformer2DModel(**TINY)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.2)
    return model.eval()


def _inputs(batch, height, width, txt_len, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 8, height, width, generator=g)
    t = torch.rand(batch, generator=g)
    ctx = torch.randn(batch, txt_len, 32, generator=g)
    return x, t, ctx


def test_padded_text_batch_matches_single_samples():
    # also pins the causal text mask: bidirectional text would let the trailing padding leak into row 1
    model = _tiny_model()
    x1, t1, c1 = _inputs(1, 4, 6, 7, seed=1)
    x2, t2, c2 = _inputs(1, 4, 6, 3, seed=2)
    with torch.no_grad():
        single1 = model(x1, t1, c1)
        single2 = model(x2, t2, c2)
        batch = model(torch.cat([x1, x2]), torch.cat([t1, t2]), torch.cat([c1, F.pad(c2, (0, 0, 0, 4))]), txt_lens=[7, 3])
    torch.testing.assert_close(batch[0], single1[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batch[1], single2[0], atol=1e-5, rtol=1e-5)


def test_gradient_checkpointing_matches_plain_backward():
    model = _tiny_model()
    x, t, ctx = _inputs(2, 4, 4, 5)
    grads = []
    for checkpointing in (False, True):
        model.gradient_checkpointing = checkpointing
        xi = x.clone().requires_grad_(True)
        model(xi, t, ctx).square().mean().backward()
        grads.append(xi.grad)
    torch.testing.assert_close(grads[0], grads[1])


def test_sigma_schedule_matches_diffusers_scheduler_config():
    from diffusers import FlowMatchEulerDiscreteScheduler

    mu = qwen_image21_utils.calculate_mu(64 * 64)
    assert mu == pytest.approx(0.6935, abs=1e-4)  # Comfy's fixed 1024x1024 shift is 0.69

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=0.9,
        base_image_seq_len=256,
        max_image_seq_len=8192,
        shift_terminal=0.02,
    )
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / 40, 40), mu=mu)
    ours = qwen_image21_utils.get_sigmas(40, mu)
    torch.testing.assert_close(ours, scheduler.sigmas.float(), atol=1e-6, rtol=1e-6)
    assert ours[-2].item() == pytest.approx(0.02, abs=1e-6)


def test_qwen21_shift_timestep_sampling_uses_unpacked_token_count():
    from musubi_tuner.qwen_image21_train_network import QwenImage21NetworkTrainer

    args = SimpleNamespace(
        timestep_sampling="qwen21_shift",
        sigmoid_scale=1.0,
        min_timestep=None,
        max_timestep=None,
        preserve_distribution_shape=False,
    )
    torch.manual_seed(0)
    latents = torch.zeros(20000, 1, 1, 64, 64)
    _, timesteps = QwenImage21NetworkTrainer().get_noisy_model_input_and_timesteps(
        args, torch.zeros_like(latents), latents, None, None, torch.device("cpu"), torch.float32
    )
    # 64x64 latents = 4096 tokens -> mu 0.6935 -> shift 2.0 -> median t = 2/3
    assert ((timesteps - 1) / 1000).median().item() == pytest.approx(2 / 3, abs=0.01)


def test_int8_convrot_embedding_returns_unrotated_rows():
    torch.manual_seed(0)
    table = torch.randn(50, 256)
    q, scale = _quantize_int8_rowwise(_convrot_hadamard(table, 256))
    emb = Int8ConvRotEmbedding(50, 256, 256, output_dtype=torch.float32)
    emb.weight = torch.nn.Parameter(q, requires_grad=False)
    emb.weight_scale = scale
    ids = torch.tensor([[3, 7, 7, 0]])
    out = emb(ids)
    assert out.shape == (1, 4, 256)
    assert ((out - table[ids]).norm() / table[ids].norm()).item() < 0.01


def _write_int8_checkpoint(model, path, group_size=64):
    sd = {}
    for key, value in model.state_dict().items():
        module = key[: -len(".weight")]
        if key.startswith("transformer_blocks.") and key.endswith(".weight") and value.ndim == 2:
            q, scale = _quantize_int8_rowwise(_convrot_hadamard(value.float(), group_size))
            sd[key] = q
            sd[f"{module}.weight_scale"] = scale
            marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size}
            sd[f"{module}.comfy_quant"] = torch.tensor(list(json.dumps(marker).encode()), dtype=torch.uint8)
        else:
            sd[key] = value.contiguous()
    save_file(sd, str(path))


def test_int8_convrot_dit_checkpoint_trains_through_frozen_int8(tmp_path):
    reference = _tiny_model()
    path = tmp_path / "tiny_int8_convrot.safetensors"
    _write_int8_checkpoint(reference, path)

    model = qwen_image21_utils.load_qwen_image21_dit(str(path), device="cpu", dtype=torch.float32, axes_dims_rope=(4, 6, 6))
    block = model.transformer_blocks[0]
    for linear in (block.attn.to_q, block.attn.to_out[0], block.img_mlp.gate_up, block.img_mlp.out):
        assert linear.weight.dtype == torch.int8 and not linear.weight.requires_grad
    assert model.img_in.weight.dtype == torch.float32

    x, t, ctx = _inputs(1, 4, 4, 5)
    with torch.no_grad():
        expected = reference(x, t, ctx)
    xi = x.clone().requires_grad_(True)
    out = model(xi, t, ctx)
    assert ((out - expected).norm() / expected.norm()).item() < 0.05  # INT8 weight + activation noise
    out.square().mean().backward()
    assert xi.grad is not None and torch.isfinite(xi.grad).all()


def test_lora_keys_are_comfy_loadable_and_presets_select_expected_layers():
    model = _tiny_model()
    comfy_keys = {"lora_unet_" + k[: -len(".weight")].replace(".", "_") for k in model.state_dict() if k.endswith(".weight")}

    network = lora_qwen_image21.create_arch_network(1.0, 4, 4, None, [], model)
    names = {lora.lora_name for lora in network.unet_loras}
    assert names <= comfy_keys
    assert "lora_unet_transformer_blocks_0_img_mlp_gate_up" in names
    assert len(names) == 6 * TINY["num_layers"] + 8  # blocks + img_in, txt_in x2, timestep x2, modulation, norm_out, proj_out

    def count(pattern):
        net = lora_qwen_image21.create_arch_network(1.0, 4, 4, None, [], _tiny_model(), include_patterns=[pattern])
        return len(net.unet_loras)

    assert count(PRESET_1) == 4 * TINY["num_layers"]
    assert count(PRESET_2) == 6 * TINY["num_layers"]
    assert count(PRESET_3) == 6 * TINY["num_layers"]


def test_vae_pads_rgb_with_opaque_alpha_and_keeps_16x_geometry():
    torch.manual_seed(0)
    vae = QwenImage21VAE(dim=8, dec_dim=8)
    with torch.no_grad():
        for p in vae.parameters():
            p.normal_(0, 0.1)
    x = torch.rand(1, 3, 32, 48) * 2 - 1
    with torch.no_grad():
        z_rgb = vae.encode(x)
        z_rgba = vae.encode(torch.cat([x, torch.ones_like(x[:, :1])], dim=1))
        decoded = vae.decode(z_rgb)
    torch.testing.assert_close(z_rgb, z_rgba)
    assert z_rgb.shape == (1, 64, 2, 3)
    assert decoded.shape == (1, 4, 32, 48)
    torch.testing.assert_close(vae.denormalize_latents(vae.normalize_latents(z_rgb)), z_rgb)


def test_text_encoder_key_mapping_keeps_only_the_text_tower():
    assert _te_key("model.layers.0.self_attn.q_proj.weight") == "layers.0.self_attn.q_proj.weight"
    assert _te_key("model.layers.0.self_attn.q_proj.weight_scale") == "layers.0.self_attn.q_proj.weight_scale"
    assert _te_key("model.embed_tokens.weight") == "embed_tokens.weight"
    assert _te_key("model.language_model.layers.3.mlp.up_proj.weight") == "layers.3.mlp.up_proj.weight"
    for skipped in (
        "lm_head.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.comfy_quant",
    ):
        assert _te_key(skipped) is None


def _peft_turbo_like(model, rank=3, seed=0):
    """PEFT/diffusers keys on the unfused gate_layer/proj pair, as the Viggle turbo LoRAs ship."""
    g = torch.Generator().manual_seed(seed)
    inner = model.inner_dim
    hidden = model.transformer_blocks[0].img_mlp.out.in_features
    shapes = {
        "modulation.1": (inner, 4 * inner),
        "transformer_blocks.0.attn.to_q": (inner, inner),
        "transformer_blocks.0.img_mlp.gate_layer": (inner, hidden),
        "transformer_blocks.0.img_mlp.proj": (inner, hidden),
        "transformer_blocks.1.img_mlp.proj": (inner, hidden),  # lone half: gate block stays zero
    }
    sd = {}
    for path, (fan_in, fan_out) in shapes.items():
        sd[f"transformer.{path}.lora_A.weight"] = torch.randn(rank, fan_in, generator=g)
        sd[f"transformer.{path}.lora_B.weight"] = torch.randn(fan_out, rank, generator=g)
    return sd


def test_peft_lora_converts_exactly_onto_the_fused_gate_up():
    model = _tiny_model()
    peft = _peft_turbo_like(model)
    converted = qwen_image21_utils.convert_lora_to_musubi(peft, default_scale=0.5)

    def delta(name):
        down, up = converted[f"lora_unet_{name}.lora_down.weight"], converted[f"lora_unet_{name}.lora_up.weight"]
        return up @ down * (converted[f"lora_unet_{name}.alpha"].item() / down.shape[0])

    def peft_delta(path):
        return peft[f"transformer.{path}.lora_B.weight"] @ peft[f"transformer.{path}.lora_A.weight"] * 0.5

    torch.testing.assert_close(delta("transformer_blocks_0_attn_to_q"), peft_delta("transformer_blocks.0.attn.to_q"))
    torch.testing.assert_close(delta("modulation_1"), peft_delta("modulation.1"))
    fused = torch.cat([peft_delta("transformer_blocks.0.img_mlp.gate_layer"), peft_delta("transformer_blocks.0.img_mlp.proj")])
    torch.testing.assert_close(delta("transformer_blocks_0_img_mlp_gate_up"), fused)
    lone = delta("transformer_blocks_1_img_mlp_gate_up")
    hidden = lone.shape[0] // 2
    assert lone[:hidden].abs().max() == 0
    torch.testing.assert_close(lone[hidden:], peft_delta("transformer_blocks.1.img_mlp.proj"))
    assert not any("gate_layer" in k or "_proj." in k for k in converted)

    musubi = {"lora_unet_x.lora_down.weight": torch.ones(1)}
    assert qwen_image21_utils.convert_lora_to_musubi(musubi) is musubi


def test_peft_lora_scale_reads_adapter_metadata(tmp_path):
    path = tmp_path / "peft.safetensors"
    config = {"transformer.lora_alpha": 128, "transformer.r": 256}
    save_file({"a": torch.zeros(1)}, str(path), metadata={"lora_adapter_metadata": json.dumps(config)})
    assert qwen_image21_utils.peft_lora_scale(str(path)) == 0.5
    save_file({"a": torch.zeros(1)}, str(path))
    assert qwen_image21_utils.peft_lora_scale(str(path)) == 1.0


def test_turbo_raw_sigmas_follow_the_viggle_contract():
    from diffusers import FlowMatchEulerDiscreteScheduler

    nodes = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
    mu = qwen_image21_utils.calculate_mu(64 * 64)
    ours = qwen_image21_utils.get_sigmas(99, mu, raw_sigmas=nodes)  # steps is ignored with raw nodes

    # Viggle's ComfyUI sigma node: exponential shift of the raw nodes, no shift_terminal, final 0
    t = torch.tensor(nodes, dtype=torch.float64)
    viggle = torch.cat([math.exp(mu) / (math.exp(mu) + (1 / t - 1)), t.new_zeros(1)]).float()
    torch.testing.assert_close(ours, viggle)

    # and the diffusers recipe: the shipped scheduler has shift_terminal = None
    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=1.0, use_dynamic_shifting=True, base_shift=0.5, max_shift=0.9, base_image_seq_len=256, max_image_seq_len=8192
    )
    scheduler.set_timesteps(sigmas=nodes, mu=mu)
    torch.testing.assert_close(ours, scheduler.sigmas.float(), atol=1e-6, rtol=1e-6)


def test_sampling_lora_attaches_unmerged_overlays_and_clears_cleanly():
    from musubi_tuner.qwen_image21_train_network import QwenImage21NetworkTrainer

    model = _tiny_model()
    x, t, ctx = _inputs(1, 4, 4, 5)
    with torch.no_grad():
        base_out = model(x, t, ctx)
    weights = qwen_image21_utils.convert_lora_to_musubi(_peft_turbo_like(model, seed=3))
    trainer = QwenImage21NetworkTrainer()
    network = lora_qwen_image21.create_arch_network_from_weights(1.0, weights, unet=model, for_inference=True)
    gate_up = model.transformer_blocks[0].img_mlp.gate_up
    weight_before = gate_up.weight.detach().clone()

    merged, attached, backups = trainer._apply_sampling_lora_network(network, weights, torch.device("cpu"))
    assert (merged, attached, backups) == (0, 4, {})  # gate_layer + proj fuse into one gate_up
    torch.testing.assert_close(gate_up.weight, weight_before)  # never merged into the base
    probe = torch.randn(2, gate_up.in_features)
    expected = F.linear(probe, weight_before) + F.linear(
        probe,
        weights["lora_unet_transformer_blocks_0_img_mlp_gate_up.lora_up.weight"]
        @ weights["lora_unet_transformer_blocks_0_img_mlp_gate_up.lora_down.weight"],
    )
    with torch.no_grad():
        error = (gate_up(probe) - expected).norm() / expected.norm()
    assert error.item() < 1e-2  # the overlay runs in bf16
    with torch.no_grad():
        assert not torch.allclose(model(x, t, ctx), base_out)

    assert trainer._clear_sampling_lora_runtime_overlays(model) == 4
    with torch.no_grad():
        torch.testing.assert_close(model(x, t, ctx), base_out)
