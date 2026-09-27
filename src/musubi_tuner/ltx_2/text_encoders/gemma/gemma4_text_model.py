"""Native Gemma-4 unified text tower for LTX-2.5 text conditioning.

LTX-2.5 ships its caption encoder as ``Gemma4UnifiedForConditionalGeneration``
(``gemma4_unified``), which no transformers release pinned by this repo can
load. The caption pipeline only consumes the text tower's 49 stacked hidden
states (embeddings + 48 layers, final one post-norm), so this module
implements exactly that subset natively. Vision/audio embedders and
generation are intentionally out of scope.

Ported against ComfyUI's ``comfy/text_encoders/gemma4.py`` (Gemma4_12B
unified) with the weight layout matching the checkpoint keys verbatim
(``model.embed_tokens.*``, ``model.layers.N.*``, ``model.norm.*``) so the
state dict loads without key mapping.

Architecture notes (verified against the ltx-2.5 checkpoint header and the
ComfyUI reference):
- 5:1 sliding/full layer pattern from ``layer_types``; sliding layers use
  head_dim 256 with 8 KV heads and full-dim RoPE (theta 1e4); full layers use
  global_head_dim 512 with 1 KV head, partial RoPE (factor 0.25, theta 1e6,
  un-rotated dims get zero frequencies) and ``attention_k_eq_v`` — V reuses
  the raw K projection (no v_proj weight exists for those layers).
- V always passes through a weightless RMS norm; Q/K have per-head-dim
  learned RMS norms; attention runs at scale=1.0 (QK-norm controls logits).
- Sandwich norms (post-attention/post-feedforward norms apply before the
  residual add) and a per-layer scalar buffer multiplying the block output.
- RMSNorm weights are stored full (no gemma3-style +1 offset).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class Gemma4TextOutput(NamedTuple):
    """Mimics the transformers output surface consumed by the base encoder."""

    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]


@dataclass
class Gemma4TextConfig:
    vocab_size: int = 262144
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    attention_k_eq_v: bool = True
    sliding_window: int = 1024
    rms_norm_eps: float = 1e-6
    rope_theta_full: float = 1_000_000.0
    rope_theta_sliding: float = 10_000.0
    partial_rotary_factor: float = 0.25
    layer_types: list[str] = field(default_factory=list)
    bos_token_id: int = 2
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if not self.layer_types:
            # 12B pattern: five sliding layers, then one full-attention layer.
            self.layer_types = [
                "full_attention" if (i % 6) == 5 else "sliding_attention" for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries for {self.num_hidden_layers} layers"
            )

    @classmethod
    def from_gemma_config(cls, gemma_config: dict) -> "Gemma4TextConfig":
        """Build from the ``gemma_config`` JSON embedded in the checkpoint metadata."""
        text = gemma_config.get("text_config", gemma_config)
        rope_parameters = text.get("rope_parameters", {})
        full_rope = rope_parameters.get("full_attention", {})
        sliding_rope = rope_parameters.get("sliding_attention", {})
        return cls(
            vocab_size=text.get("vocab_size", 262144),
            hidden_size=text.get("hidden_size", 3840),
            intermediate_size=text.get("intermediate_size", 15360),
            num_hidden_layers=text.get("num_hidden_layers", 48),
            num_attention_heads=text.get("num_attention_heads", 16),
            num_key_value_heads=text.get("num_key_value_heads", 8),
            num_global_key_value_heads=text.get("num_global_key_value_heads", 1),
            head_dim=text.get("head_dim", 256),
            global_head_dim=text.get("global_head_dim", 512),
            attention_k_eq_v=text.get("attention_k_eq_v", True),
            sliding_window=text.get("sliding_window", 1024),
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            rope_theta_full=full_rope.get("rope_theta", 1_000_000.0),
            rope_theta_sliding=sliding_rope.get("rope_theta", 10_000.0),
            partial_rotary_factor=full_rope.get("partial_rotary_factor", 0.25),
            layer_types=list(text.get("layer_types", [])),
            bos_token_id=text.get("bos_token_id", 2),
            pad_token_id=text.get("pad_token_id", 0),
        )


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), weight=self.weight, eps=self.eps)


def _apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Half-split convention; un-rotated dims carry zero frequencies (cos=1, sin=0).
    half = x.shape[-1] // 2
    out = x * cos
    out[..., :half] -= x[..., half:] * sin[..., :half]
    out[..., half:] += x[..., :half] * sin[..., half:]
    return out


class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, *, sliding: bool) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        head_dim = config.head_dim if sliding else config.global_head_dim
        self.head_dim = head_dim
        k_eq_v = config.attention_k_eq_v and not sliding
        self.num_kv_heads = config.num_key_value_heads if sliding else config.num_global_key_value_heads
        inner = self.num_heads * head_dim
        kv_inner = self.num_kv_heads * head_dim
        self.eps = config.rms_norm_eps

        self.q_proj = nn.Linear(config.hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_inner, bias=False)
        self.v_proj = None if k_eq_v else nn.Linear(config.hidden_size, kv_inner, bias=False)
        self.o_proj = nn.Linear(inner, config.hidden_size, bias=False)
        self.q_norm = Gemma4RMSNorm(head_dim, config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(head_dim, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        # k_eq_v: V is the raw K projection, before k_norm and RoPE.
        v = k if self.v_proj is None else self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = F.rms_norm(v, (self.head_dim,), eps=self.eps).transpose(1, 2)
        q = _apply_rotary_pos_emb(q, cos, sin)
        k = _apply_rotary_pos_emb(k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # QK-norm controls logit magnitude; the reference runs unscaled attention.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Gemma4MLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_type: str) -> None:
        super().__init__()
        self.sliding = layer_type == "sliding_attention"
        self.self_attn = Gemma4Attention(config, sliding=self.sliding)
        self.mlp = Gemma4MLP(config)
        eps = config.rms_norm_eps
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps)
        self.register_buffer("layer_scalar", torch.empty(1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cos, sin)
        x = residual + self.post_attention_layernorm(h)

        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        x = residual + self.post_feedforward_layernorm(h)

        return x * self.layer_scalar.to(x.dtype)


def _global_inv_freq(config: Gemma4TextConfig, device: torch.device | None = None) -> torch.Tensor:
    # Un-rotated global dims get zero inverse frequency -> cos 1 / sin 0 (identity).
    rope_angles = int(config.partial_rotary_factor * config.global_head_dim // 2)
    inv = 1.0 / (
        config.rope_theta_full
        ** (torch.arange(0, 2 * rope_angles, 2, device=device).float() / config.global_head_dim)
    )
    nope = config.global_head_dim // 2 - rope_angles
    if nope > 0:
        inv = torch.cat([inv, torch.zeros(nope, device=device)])
    return inv


def _sliding_inv_freq(config: Gemma4TextConfig, device: torch.device | None = None) -> torch.Tensor:
    return 1.0 / (
        config.rope_theta_sliding ** (torch.arange(0, config.head_dim, 2, device=device).float() / config.head_dim)
    )


class _Gemma4Stack(nn.Module):
    """Serialized under the ``model.`` prefix to match checkpoint keys."""

    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Gemma4DecoderLayer(config, layer_type) for layer_type in config.layer_types
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.register_buffer("_global_inv_freq", _global_inv_freq(config), persistent=False)
        self.register_buffer("_sliding_inv_freq", _sliding_inv_freq(config), persistent=False)

    @staticmethod
    def _freqs(inv_freq: torch.Tensor, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions[:, :, None].float() * inv_freq[None, None, :].float()
        emb = torch.cat((freqs, freqs), dim=-1)
        # [B, 1, S, D] broadcasting over heads.
        return emb.cos().unsqueeze(1).to(dtype), emb.sin().unsqueeze(1).to(dtype)

    def _build_masks(
        self, attention_mask: torch.Tensor | None, batch: int, seq: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        min_val = torch.finfo(dtype).min
        mask = None
        if attention_mask is not None:
            mask = (1.0 - attention_mask.to(dtype)).reshape(batch, 1, 1, seq).expand(batch, 1, seq, seq)
            mask = mask.masked_fill(mask.bool(), min_val).clone()
        if seq > 1:
            causal = torch.zeros(seq, seq, dtype=dtype, device=device)
            causal.masked_fill_(torch.ones_like(causal, dtype=torch.bool).triu_(1), min_val)
            mask = causal if mask is None else mask + causal
        sliding_mask = mask
        if seq > self.config.sliding_window and any(layer.sliding for layer in self.layers):
            window = torch.zeros(seq, seq, dtype=dtype, device=device)
            window.masked_fill_(torch.ones_like(window, dtype=torch.bool).tril_(-self.config.sliding_window), min_val)
            sliding_mask = window if mask is None else mask + window
        return mask, sliding_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        batch, seq = input_ids.shape
        x = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)
        positions = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        global_cos, global_sin = self._freqs(self._global_inv_freq, positions, x.dtype)
        sliding_cos, sliding_sin = self._freqs(self._sliding_inv_freq, positions, x.dtype)
        full_mask, sliding_mask = self._build_masks(attention_mask, batch, seq, x.device, x.dtype)

        hidden_states = [x]
        for i, layer in enumerate(self.layers):
            if layer.sliding:
                x = layer(x, sliding_mask, sliding_cos, sliding_sin)
            else:
                x = layer(x, full_mask, global_cos, global_sin)
            if i < len(self.layers) - 1:
                hidden_states.append(x)
        x = self.norm(x)
        # transformers convention: final entry is the post-norm last layer output.
        hidden_states.append(x)
        return tuple(hidden_states)


class Gemma4TextEncoder(nn.Module):
    """Drop-in for the ``model`` attribute of GemmaTextEncoderModelBase.

    Exposes the call signature and output surface the base encoder uses from
    ``Gemma3ForConditionalGeneration`` (``output_hidden_states=True`` path,
    ``get_input_embeddings``); generation/enhancement is not supported.
    """

    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = config
        self.model = _Gemma4Stack(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        **_: object,
    ) -> Gemma4TextOutput:
        if not output_hidden_states:
            raise ValueError("Gemma4TextEncoder only supports output_hidden_states=True")
        with torch.no_grad():
            hidden_states = self.model(input_ids, attention_mask)
        return Gemma4TextOutput(last_hidden_state=hidden_states[-1], hidden_states=hidden_states)

    def generate(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Prompt enhancement/generation is not implemented for the native Gemma-4 encoder; "
            "it is only used to produce LTX-2.5 caption embeddings."
        )


def gemma4_metadata_config(safetensors_path: str) -> dict | None:
    """Return the parsed ``gemma_config`` metadata if the file is a Gemma-4 checkpoint."""
    from safetensors import safe_open

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    raw = metadata.get("gemma_config")
    if raw is None:
        return None
    config = json.loads(raw)
    if "gemma4" not in str(config.get("model_type", "")):
        return None
    return config


def extract_tokenizer_json_bytes(safetensors_path: str) -> bytes:
    """Extract the HF-tokenizers JSON embedded as a U8 tensor in the checkpoint."""
    from safetensors import safe_open

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        if "tokenizer_json" not in f.keys():
            raise ValueError(
                f"No 'tokenizer_json' tensor found in {safetensors_path}; "
                "cannot build the Gemma-4 tokenizer from this file."
            )
        return f.get_tensor("tokenizer_json").numpy().tobytes()


def _scan_int8_convrot_layers(safetensors_path: str) -> dict:
    """Parse ``.comfy_quant`` markers under ``model.`` (comfy INT8 ConvRot layout)."""
    from safetensors import safe_open

    from musubi_tuner.modules.int8_optimization_utils import Int8ConvRotConfig

    layers: dict[str, Int8ConvRotConfig] = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for marker_key in (k for k in keys if k.startswith("model.") and k.endswith(".comfy_quant")):
            module_name = marker_key[: -len(".comfy_quant")]
            marker = json.loads(bytes(f.get_tensor(marker_key).tolist()).decode("utf-8"))
            params = marker.get("params", {})
            if marker.get("format") != "int8_tensorwise" or not marker.get("convrot", params.get("convrot", False)):
                raise ValueError(
                    f"Unsupported quantization format for {module_name}: {marker.get('format')!r} "
                    "(only int8 ConvRot is supported; nvfp4 files are not)"
                )
            scale_key = f"{module_name}.weight_scale"
            if scale_key not in keys:
                raise ValueError(f"Quantized layer {module_name} is missing its weight_scale tensor")
            layers[module_name] = Int8ConvRotConfig(
                group_size=int(marker.get("convrot_groupsize", params.get("convrot_groupsize", 256))),
                scale_shape=tuple(f.get_slice(scale_key).get_shape()),
            )
    return layers


def load_gemma4_text_encoder(
    safetensors_path: str,
    gemma_config: dict,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> Gemma4TextEncoder:
    """Build the native Gemma-4 text tower and stream weights from the checkpoint.

    Supports both the bf16 release file and the comfy INT8 ConvRot variant
    (per-linear I8 weight + F32 per-channel scale + ``.comfy_quant`` marker);
    quantized linears run through comfy-kitchen kernels when available.

    Only ``model.*`` tensors are consumed; the unified checkpoint's vision/audio
    embedders, aggregate projections, and embedded tokenizer assets are ignored
    here (projections load via the single-file checkpoint, the tokenizer via
    :func:`extract_tokenizer_json_bytes`).
    """
    from safetensors import safe_open

    from musubi_tuner.modules.int8_optimization_utils import apply_int8_convrot_monkey_patch

    config = Gemma4TextConfig.from_gemma_config(gemma_config)
    load_device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # Markers use "model." paths relative to the checkpoint root; the encoder's
    # stack lives under the same "model." attribute, so names map 1:1.
    int8_layers = _scan_int8_convrot_layers(safetensors_path)

    with torch.device("meta"):
        encoder = Gemma4TextEncoder(config)
    if int8_layers:
        apply_int8_convrot_monkey_patch(encoder, int8_layers)
        logger.info("Loading INT8 ConvRot Gemma-4 text tower (%d quantized Linears)", len(int8_layers))

    int8_weight_keys = {f"{name}.weight" for name in int8_layers}
    expected = {name for name, _ in encoder.named_parameters()}
    expected |= {name for name, buf in encoder.named_buffers() if "_inv_freq" not in name}

    loaded: set[str] = set()
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if not key.startswith("model.") or key.endswith(".comfy_quant"):
                continue
            if key.endswith(".weight_scale_2"):
                raise ValueError(f"Unsupported quantization tensor {key}; only int8 ConvRot files are supported")
            if key not in expected:
                raise ValueError(f"Unexpected Gemma-4 text-tower tensor: {key}")
            tensor = f.get_tensor(key)
            module_path, _, attr = key.rpartition(".")
            submodule = encoder.get_submodule(module_path)
            reference = getattr(submodule, attr)
            if reference.shape != tensor.shape:
                raise ValueError(f"Shape mismatch for {key}: model {tuple(reference.shape)} vs checkpoint {tuple(tensor.shape)}")
            if key in int8_weight_keys or key.endswith(".weight_scale"):
                # Quantized payloads keep their checkpoint dtype (I8 / F32 scale).
                value = tensor.to(device=load_device)
            else:
                value = tensor.to(device=load_device, dtype=torch_dtype)
            if isinstance(reference, nn.Parameter):
                setattr(submodule, attr, nn.Parameter(value, requires_grad=False))
            else:
                setattr(submodule, attr, value)
            loaded.add(key)

    missing = expected - loaded
    if missing:
        raise ValueError(f"Gemma-4 checkpoint is missing {len(missing)} text-tower tensors, e.g. {sorted(missing)[:5]}")

    # The non-persistent rope buffers were meta-initialized; recompute them on the load device.
    stack = encoder.model
    stack._global_inv_freq = _global_inv_freq(config, device=load_device)
    stack._sliding_inv_freq = _sliding_inv_freq(config, device=load_device)
    return encoder.eval()
