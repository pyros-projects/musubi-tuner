"""Tests for the native Gemma-4 text encoder (LTX-2.5 caption pipeline)."""

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from musubi_tuner.ltx_2.text_encoders.gemma.gemma4_text_model import (  # noqa: E402
    Gemma4TextConfig,
    Gemma4TextEncoder,
    extract_tokenizer_json_bytes,
    gemma4_metadata_config,
    load_gemma4_text_encoder,
)
from musubi_tuner.ltx_2.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer  # noqa: E402


def _tiny_config(**overrides) -> Gemma4TextConfig:
    defaults = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        attention_k_eq_v=True,
        sliding_window=1024,
        layer_types=["sliding_attention", "full_attention"],
    )
    defaults.update(overrides)
    return Gemma4TextConfig(**defaults)


def _init_weights(encoder: Gemma4TextEncoder, seed: int = 0) -> None:
    generator = torch.Generator().manual_seed(seed)
    for _, param in encoder.named_parameters():
        param.data = torch.randn(param.shape, generator=generator, dtype=torch.float32) * 0.05
    for name, buf in encoder.named_buffers():
        if name.endswith("layer_scalar"):
            buf.fill_(1.0)


class TestGemma4Model:
    def test_state_dict_layout_matches_checkpoint_keys(self):
        encoder = Gemma4TextEncoder(_tiny_config())
        keys = set(encoder.state_dict().keys())

        assert "model.embed_tokens.weight" in keys
        assert "model.norm.weight" in keys
        # Sliding layer (0) has a v_proj; global layer (1) does not (attention_k_eq_v).
        assert "model.layers.0.self_attn.v_proj.weight" in keys
        assert "model.layers.1.self_attn.v_proj.weight" not in keys
        for i in (0, 1):
            for name in (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "layer_scalar",
            ):
                assert f"model.layers.{i}.{name}" in keys
        # Non-persistent rope buffers must not be serialized.
        assert not any("_inv_freq" in k for k in keys)

    def test_default_layer_pattern_is_five_to_one(self):
        config = Gemma4TextConfig(num_hidden_layers=12, layer_types=[])
        assert config.layer_types[5] == "full_attention"
        assert config.layer_types[11] == "full_attention"
        assert config.layer_types.count("full_attention") == 2

    def test_forward_hidden_states_count_and_shapes(self):
        config = _tiny_config()
        encoder = Gemma4TextEncoder(config)
        _init_weights(encoder)
        input_ids = torch.randint(0, config.vocab_size, (2, 6))
        out = encoder(input_ids=input_ids, attention_mask=torch.ones(2, 6, dtype=torch.long))

        assert len(out.hidden_states) == config.num_hidden_layers + 1
        for state in out.hidden_states:
            assert state.shape == (2, 6, config.hidden_size)
        # Final entry is post-norm and must equal last_hidden_state.
        assert torch.equal(out.hidden_states[-1], out.last_hidden_state)
        assert not torch.isnan(out.last_hidden_state).any()

    def test_padded_tokens_do_not_influence_real_tokens(self):
        config = _tiny_config()
        encoder = Gemma4TextEncoder(config)
        _init_weights(encoder)

        ids_a = torch.tensor([[0, 0, 5, 6, 7]])
        ids_b = torch.tensor([[9, 3, 5, 6, 7]])  # different content in padded slots
        mask = torch.tensor([[0, 0, 1, 1, 1]])

        out_a = encoder(input_ids=ids_a, attention_mask=mask)
        out_b = encoder(input_ids=ids_b, attention_mask=mask)
        # Real-token positions must be identical; padded positions are dropped downstream.
        torch.testing.assert_close(
            out_a.last_hidden_state[:, 2:], out_b.last_hidden_state[:, 2:], rtol=0, atol=0
        )

    def test_causality(self):
        config = _tiny_config()
        encoder = Gemma4TextEncoder(config)
        _init_weights(encoder)

        ids_a = torch.tensor([[1, 2, 3, 4]])
        ids_b = torch.tensor([[1, 2, 3, 9]])
        mask = torch.ones(1, 4, dtype=torch.long)
        out_a = encoder(input_ids=ids_a, attention_mask=mask)
        out_b = encoder(input_ids=ids_b, attention_mask=mask)
        torch.testing.assert_close(
            out_a.last_hidden_state[:, :3], out_b.last_hidden_state[:, :3], rtol=0, atol=0
        )

    def test_sliding_window_mask_restricts_attention(self):
        base = _tiny_config(sliding_window=2)
        wide = _tiny_config(sliding_window=1024)
        narrow_enc = Gemma4TextEncoder(base)
        wide_enc = Gemma4TextEncoder(wide)
        _init_weights(narrow_enc)
        wide_enc.load_state_dict(narrow_enc.state_dict())

        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mask = torch.ones(1, 6, dtype=torch.long)
        out_narrow = narrow_enc(input_ids=ids, attention_mask=mask)
        out_wide = wide_enc(input_ids=ids, attention_mask=mask)
        assert not torch.allclose(out_narrow.last_hidden_state, out_wide.last_hidden_state)

    def test_generate_is_unsupported(self):
        encoder = Gemma4TextEncoder(_tiny_config())
        with pytest.raises(NotImplementedError):
            encoder.generate()


def _write_checkpoint(path: Path, encoder: Gemma4TextEncoder, config: Gemma4TextConfig, extra=None) -> None:
    from safetensors.torch import save_file

    tensors = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in encoder.state_dict().items()}
    tensors["text_embedding_projection.video_aggregate_embed.weight"] = torch.zeros(4, 8, dtype=torch.bfloat16)
    tensors["vision_model.patch_dense.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)
    tensors["tokenizer_json"] = torch.frombuffer(bytearray(b'{"not": "a real tokenizer"}'), dtype=torch.uint8).clone()
    if extra:
        tensors.update(extra)
    gemma_config = {
        "model_type": "gemma4_unified",
        "text_config": {
            "model_type": "gemma4_unified_text",
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "num_global_key_value_heads": config.num_global_key_value_heads,
            "head_dim": config.head_dim,
            "global_head_dim": config.global_head_dim,
            "attention_k_eq_v": config.attention_k_eq_v,
            "sliding_window": config.sliding_window,
            "layer_types": config.layer_types,
            "rope_parameters": {
                "full_attention": {"rope_theta": config.rope_theta_full, "partial_rotary_factor": config.partial_rotary_factor},
                "sliding_attention": {"rope_theta": config.rope_theta_sliding},
            },
        },
    }
    save_file(tensors, str(path), metadata={"gemma_config": json.dumps(gemma_config)})


class TestGemma4Loading:
    def test_metadata_detection(self, tmp_path):
        config = _tiny_config()
        encoder = Gemma4TextEncoder(config)
        _init_weights(encoder)
        path = tmp_path / "gemma4.safetensors"
        _write_checkpoint(path, encoder, config)

        detected = gemma4_metadata_config(str(path))
        assert detected is not None
        assert detected["text_config"]["hidden_size"] == config.hidden_size

    def test_non_gemma4_returns_none(self, tmp_path):
        from safetensors.torch import save_file

        path = tmp_path / "other.safetensors"
        save_file({"w": torch.zeros(1)}, str(path), metadata={"other": "x"})
        assert gemma4_metadata_config(str(path)) is None

    def test_load_roundtrip_matches_reference(self, tmp_path):
        config = _tiny_config()
        reference = Gemma4TextEncoder(config)
        _init_weights(reference, seed=7)
        path = tmp_path / "gemma4.safetensors"
        _write_checkpoint(path, reference, config)

        loaded = load_gemma4_text_encoder(
            str(path), gemma4_metadata_config(str(path)), torch_dtype=torch.float32, device=torch.device("cpu")
        )
        ids = torch.randint(0, config.vocab_size, (1, 5))
        mask = torch.ones(1, 5, dtype=torch.long)
        out_ref = reference.to(torch.float32)(input_ids=ids, attention_mask=mask)
        out_loaded = loaded(input_ids=ids, attention_mask=mask)
        # bf16 round-trip through the checkpoint bounds the tolerance.
        torch.testing.assert_close(out_loaded.last_hidden_state, out_ref.last_hidden_state, rtol=0.05, atol=0.05)
        assert len(out_loaded.hidden_states) == config.num_hidden_layers + 1

    def test_missing_tensor_fails_closed(self, tmp_path):
        from safetensors import safe_open
        from safetensors.torch import save_file

        config = _tiny_config()
        encoder = Gemma4TextEncoder(config)
        _init_weights(encoder)
        path = tmp_path / "gemma4.safetensors"
        _write_checkpoint(path, encoder, config)

        with safe_open(str(path), framework="pt", device="cpu") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys() if k != "model.norm.weight"}
            metadata = f.metadata()
        broken = tmp_path / "broken.safetensors"
        save_file(tensors, str(broken), metadata=metadata)

        with pytest.raises(ValueError, match="missing"):
            load_gemma4_text_encoder(str(broken), gemma4_metadata_config(str(broken)), device=torch.device("cpu"))

    def test_unsupported_quant_format_rejected(self, tmp_path):
        config = _tiny_config()
        encoder = Gemma4TextEncoder(config)
        _init_weights(encoder)
        path = tmp_path / "quant.safetensors"
        marker = torch.frombuffer(bytearray(json.dumps({"format": "nvfp4"}).encode()), dtype=torch.uint8).clone()
        _write_checkpoint(
            path, encoder, config, extra={"model.layers.0.self_attn.q_proj.comfy_quant": marker}
        )
        with pytest.raises(ValueError, match="int8 ConvRot"):
            load_gemma4_text_encoder(str(path), gemma4_metadata_config(str(path)), device=torch.device("cpu"))

    def test_extract_tokenizer_json_bytes(self, tmp_path):
        config = _tiny_config()
        encoder = Gemma4TextEncoder(config)
        _init_weights(encoder)
        path = tmp_path / "gemma4.safetensors"
        _write_checkpoint(path, encoder, config)
        assert extract_tokenizer_json_bytes(str(path)) == b'{"not": "a real tokenizer"}'


class TestGemma4Int8ConvRot:
    GROUP = 4  # power of 4, divides hidden_size=16 and intermediate=32

    def _quantize_linear(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce (int8 rotated weight, per-channel scale) whose dense equivalent
        is W_dense = (W_int8 * scale) @ H — matching the ConvRot forward x -> W_rot·(H·x)."""
        from musubi_tuner.modules.int8_optimization_utils import _convrot_hadamard

        w_rot = _convrot_hadamard(weight.float(), self.GROUP)
        scale = (w_rot.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-30)
        w_int8 = (w_rot / scale).round().clamp(-128, 127).to(torch.int8)
        return w_int8, scale.float()

    def _dense_equivalent(self, w_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        from musubi_tuner.modules.int8_optimization_utils import _convrot_hadamard

        # H is symmetric orthogonal, so rotating the rows back equals @ H.
        return _convrot_hadamard(w_int8.float() * scale, self.GROUP)

    def test_int8_checkpoint_loads_and_matches_dense_reference(self, tmp_path):
        from safetensors.torch import save_file

        config = _tiny_config()
        encoder = Gemma4TextEncoder(config)
        _init_weights(encoder, seed=11)

        marker = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": self.GROUP})
        marker_tensor = torch.frombuffer(bytearray(marker.encode()), dtype=torch.uint8).clone()

        tensors = {}
        reference = Gemma4TextEncoder(config)
        reference.load_state_dict(encoder.state_dict())
        for key, value in encoder.state_dict().items():
            is_proj = key.endswith(".weight") and (".self_attn." in key or ".mlp." in key) and "norm" not in key
            if is_proj:
                w_int8, scale = self._quantize_linear(value)
                tensors[key] = w_int8
                tensors[key[: -len(".weight")] + ".weight_scale"] = scale
                tensors[key[: -len(".weight")] + ".comfy_quant"] = marker_tensor.clone()
                # Reference model gets the mathematically equivalent dense weight.
                module_path, _, attr = key.rpartition(".")
                submodule = reference.get_submodule(module_path)
                setattr(submodule, attr, torch.nn.Parameter(self._dense_equivalent(w_int8, scale), requires_grad=False))
            else:
                tensors[key] = value.detach().to(torch.float32).contiguous()
        tensors["tokenizer_json"] = torch.frombuffer(bytearray(b"{}"), dtype=torch.uint8).clone()

        gemma_config = json.loads(json.dumps({"model_type": "gemma4_unified", "text_config": {
            "vocab_size": config.vocab_size, "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size, "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads, "num_key_value_heads": config.num_key_value_heads,
            "num_global_key_value_heads": config.num_global_key_value_heads, "head_dim": config.head_dim,
            "global_head_dim": config.global_head_dim, "attention_k_eq_v": config.attention_k_eq_v,
            "sliding_window": config.sliding_window, "layer_types": config.layer_types,
        }}))
        path = tmp_path / "gemma4-int8.safetensors"
        save_file(tensors, str(path), metadata={"gemma_config": json.dumps(gemma_config)})

        loaded = load_gemma4_text_encoder(
            str(path), gemma4_metadata_config(str(path)), torch_dtype=torch.float32, device=torch.device("cpu")
        )
        # Quantized linears keep int8 payloads + scales.
        q0 = loaded.model.layers[0].self_attn.q_proj
        assert q0.weight.dtype == torch.int8
        assert q0.weight_scale.dtype == torch.float32

        ids = torch.randint(0, config.vocab_size, (1, 6))
        mask = torch.ones(1, 6, dtype=torch.long)
        out_int8 = loaded(input_ids=ids, attention_mask=mask)
        out_ref = reference.to(torch.float32)(input_ids=ids, attention_mask=mask)
        # Only input-activation quantization separates the two (weights are
        # bit-identical after dequant+rotation); tolerance covers that.
        torch.testing.assert_close(out_int8.last_hidden_state, out_ref.last_hidden_state, rtol=0.08, atol=0.08)
        assert len(out_int8.hidden_states) == config.num_hidden_layers + 1

    def test_missing_scale_fails_closed(self, tmp_path):
        config = _tiny_config()
        encoder = Gemma4TextEncoder(config)
        _init_weights(encoder)
        marker = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": self.GROUP})
        path = tmp_path / "broken-int8.safetensors"
        _write_checkpoint(
            path, encoder, config,
            extra={"model.layers.0.self_attn.q_proj.comfy_quant": torch.frombuffer(bytearray(marker.encode()), dtype=torch.uint8).clone()},
        )
        with pytest.raises(ValueError, match="weight_scale"):
            load_gemma4_text_encoder(str(path), gemma4_metadata_config(str(path)), device=torch.device("cpu"))


class TestGemma4ModuleOpsIntegration:
    def test_module_ops_load_native_gemma4_and_tokenizer(self, tmp_path, monkeypatch):
        """The real entry point detects gemma4, loads the native encoder and the embedded tokenizer."""
        import torch.nn as nn

        from musubi_tuner.ltx_2.text_encoders.gemma.encoders.base_encoder import (
            GemmaTextEncoderModelBase,
            module_ops_from_gemma_root,
        )

        config = _tiny_config()
        reference = Gemma4TextEncoder(config)
        _init_weights(reference)
        path = tmp_path / "gemma4.safetensors"
        _write_checkpoint(path, reference, config)
        # Overwrite the placeholder with a real tokenizers JSON so the tokenizer loads.
        from safetensors import safe_open
        from safetensors.torch import save_file

        with safe_open(str(path), framework="pt", device="cpu") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys() if k != "tokenizer_json"}
            metadata = f.metadata()
        json_bytes = TestGemma4Tokenizer()._tokenizer_json_bytes()
        tensors["tokenizer_json"] = torch.frombuffer(bytearray(json_bytes), dtype=torch.uint8).clone()
        save_file(tensors, str(path), metadata=metadata)

        monkeypatch.setenv("LTX2_GEMMA_MAX_LENGTH", "6")
        gemma_ops, tokenizer_ops = module_ops_from_gemma_root(
            None, gemma_safetensors=str(path), torch_dtype=torch.float32, device=torch.device("cpu")
        )
        module = GemmaTextEncoderModelBase(feature_extractor_linear=nn.Identity())
        module = gemma_ops.mutator(module)
        module = tokenizer_ops.mutator(module)

        assert isinstance(module.model, Gemma4TextEncoder)
        pairs = module.tokenizer.tokenize_with_weights("hello world")["gemma"]
        input_ids = torch.tensor([[t for t, _ in pairs]])
        attention_mask = torch.tensor([[w for _, w in pairs]])
        out = module.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        assert len(out.hidden_states) == config.num_hidden_layers + 1
        assert not torch.isnan(out.last_hidden_state[:, -3:]).any()


class TestGemma4Tokenizer:
    def _tokenizer_json_bytes(self) -> bytes:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "hello": 3, "world": 4, "[UNK]": 5}
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        return tokenizer.to_str().encode("utf-8")

    def test_json_bytes_select_tokenizers_backend_with_bos_and_left_padding(self):
        wrapper = LTXVGemmaTokenizer(self._tokenizer_json_bytes(), max_length=6)
        pairs = wrapper.tokenize_with_weights("hello world")["gemma"]
        assert len(pairs) == 6
        ids = [t for t, _ in pairs]
        attn = [w for _, w in pairs]
        # Left padding, then BOS (prepended by the adapter), then the tokens.
        assert ids == [0, 0, 0, 2, 3, 4]
        assert attn == [0, 0, 0, 1, 1, 1]

    def test_truncation(self):
        wrapper = LTXVGemmaTokenizer(self._tokenizer_json_bytes(), max_length=2)
        pairs = wrapper.tokenize_with_weights("hello world hello world")["gemma"]
        assert len(pairs) == 2
        assert all(w == 1 for _, w in pairs)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        pytest.main([__file__, "-v", "--basetemp", td])
