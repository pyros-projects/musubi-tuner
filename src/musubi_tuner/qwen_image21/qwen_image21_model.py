"""Qwen-Image 2.1 DiT (QwenImage21Transformer2DModel) for LoRA training.

Pure-PyTorch port of the text-to-image path of ComfyUI's ``comfy/ldm/qwen_image21/model.py``
(its ``in_training`` branch), which follows diffusers' ``transformer_qwenimage21.py``.
Module names match the checkpoint keys 1:1, so Comfy files load strictly and musubi's
``lora_unet_*`` LoRA keys resolve in ComfyUI without conversion.

One stream: text tokens first (causal among themselves, modulated from t = 0), then the
target image tokens (full attention over text + image, modulated from the sampled t).
Reference images (edit mode) and Comfy's prefix KV cache are not ported.
"""

import math
from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """(..., N) positions -> (..., N, dim // 2, 2, 2) rotation matrices (flux/Comfy layout)."""
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (theta**scale)
    out = pos.to(torch.float64).unsqueeze(-1) * omega
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x: (B, N, H, D); freqs: (B or 1, N, 1, D // 2, 2, 2)."""
    x_ = x.to(freqs.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
    return out.reshape(*x.shape).type_as(x)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> torch.Tensor:
    # flux convention: t in [0, 1] is scaled by 1000, cos half first
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    # affine-free LayerNorm; cast back so autocast's fp32 output does not leak into the INT8 linears
    return F.layer_norm(x, (x.shape[-1],), eps=eps).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, weight=self.weight.to(x.dtype), eps=self.eps).to(x.dtype)


class ZeroCenteredRMSNorm(nn.Module):
    # stored weight is scale - 1, applied in fp32
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.float() + 1.0
        return F.rms_norm(x.float(), w.shape, weight=w, eps=self.eps).to(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(in_dim, eps=eps)
        self.in_layer = nn.Linear(in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=False)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.timestep_embedder(timestep_embedding(timestep.float(), 256).to(dtype))


class SwiGLUFeedForward(nn.Module):
    # [gate; up] fused in one Linear, as in Comfy files (LoRAs address the fused weight)
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.out(F.silu(gate) * up)


def text_image_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, txt_len: int, key_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """Block-causal attention over [text, image]: text is causal, image attends to everything.

    q, k, v: (B, N, H, D). key_mask: (B, N) bool marking valid keys (text padding False), or None.
    Causal text queries never reach the trailing text padding, so only the image queries need the mask.
    """
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    txt = F.scaled_dot_product_attention(q[:, :, :txt_len], k[:, :, :txt_len], v[:, :, :txt_len], is_causal=True)
    mask = None if key_mask is None else key_mask[:, None, None, :]
    img = F.scaled_dot_product_attention(q[:, :, txt_len:], k, v, attn_mask=mask)
    return torch.cat([txt, img], dim=2).transpose(1, 2).flatten(2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=False)])
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

    def forward(self, x: torch.Tensor, pe: torch.Tensor, attn_fn) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.to_q(x).view(B, N, self.heads, -1)
        k = self.to_k(x).view(B, N, self.heads, -1)
        v = self.to_v(x).view(B, N, self.heads, -1)
        q = apply_rope(self.norm_q(q), pe)
        k = apply_rope(self.norm_k(k), pe)
        return self.to_out[0](attn_fn(q, k, v))


def _split_rows(p: torch.Tensor):
    # shared modulation rows: (t = 0 row for the text prefix, sampled-t rows for the target)
    return p[-1:].unsqueeze(1), p[:-1].unsqueeze(1)


def _modulate(x: torch.Tensor, scale, prefix_len: int) -> torch.Tensor:
    s_prefix, s_target = scale
    return torch.cat([x[:, :prefix_len] * (1 + s_prefix), x[:, prefix_len:] * (1 + s_target)], dim=1)


def _gate(y: torch.Tensor, gate, prefix_len: int) -> torch.Tensor:
    g_prefix, g_target = gate
    return torch.cat([y[:, :prefix_len] * g_prefix, y[:, prefix_len:] * g_target], dim=1)


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: int = 3, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.attn = Attention(dim, num_attention_heads, attention_head_dim, eps=eps)
        self.img_mlp = SwiGLUFeedForward(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor, mod, pe: torch.Tensor, attn_fn, prefix_len: int) -> torch.Tensor:
        scale1, gate1, scale2, gate2 = mod
        x = x + _gate(self.attn(_modulate(_layer_norm(x, self.eps), scale1, prefix_len), pe, attn_fn), gate1, prefix_len)
        x = x + _gate(self.img_mlp(_modulate(_layer_norm(x, self.eps), scale2, prefix_len)), gate2, prefix_len)
        return x


class LastLayer(nn.Module):
    # scale only, no shift
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.eps = eps

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale = self.linear(F.silu(temb)).unsqueeze(1)
        return _layer_norm(x, self.eps) * (1 + scale)


class QwenImage21Transformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        context_in_dim: int = 4096,
        mlp_ratio: int = 3,
        axes_dims_rope: Sequence[int] = (16, 56, 56),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.axes_dims_rope = tuple(axes_dims_rope)
        self.rope_theta = 10000

        self.time_text_embed = TimestepProjEmbeddings(self.inner_dim)
        self.txt_in = TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels, self.inner_dim, bias=False)
        # one modulation shared by every block
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, eps=eps)
                for _ in range(num_layers)
            ]
        )
        self.norm_out = LastLayer(self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, out_channels, bias=False)

        self.gradient_checkpointing = False

    @property
    def device(self) -> torch.device:
        return self.img_in.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.img_in.weight.dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        if cpu_offload:
            raise ValueError("Qwen-Image 2.1 does not support --gradient_checkpointing_cpu_offload")
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    # the base trainer calls these around sampling; there is no block swap to switch
    def switch_block_swap_for_inference(self):
        pass

    def switch_block_swap_for_training(self):
        pass

    def rope_freqs(self, txt_lens: Sequence[int], txt_len: int, height: int, width: int, device) -> torch.Tensor:
        """(B or 1, txt_len + H*W, 1, head_dim // 2, 2, 2) rotary tables.

        Text token i sits at (i, i, i). Image tokens sit at t = the sample's real text length, with a
        (row, column) grid centred on zero, exactly as Comfy's build_sequence lays out the target.
        """
        # one shared row when every sample has the same text length, else one row per sample
        starts = [txt_lens[0]] if len(set(txt_lens)) == 1 else list(txt_lens)
        hh = torch.arange(height, device=device, dtype=torch.float32) - (height - height // 2)
        ww = torch.arange(width, device=device, dtype=torch.float32) - (width - width // 2)
        ids = torch.zeros(len(starts), txt_len + height * width, 3, device=device)
        ids[:, :txt_len] = torch.arange(txt_len, device=device, dtype=torch.float32)[None, :, None]
        ids[:, txt_len:, 0] = torch.tensor(starts, device=device, dtype=torch.float32)[:, None]
        ids[:, txt_len:, 1] = hh[:, None].expand(height, width).flatten()
        ids[:, txt_len:, 2] = ww[None, :].expand(height, width).flatten()
        freqs = torch.cat([rope(ids[..., i], self.axes_dims_rope[i], self.rope_theta) for i in range(3)], dim=-3)
        return freqs.unsqueeze(2)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        txt_lens: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """x: (B, 64, H, W) normalized latents; timesteps: (B,) sigma in [0, 1];
        context: (B, L, 4096) text states, right-padded to L with ``txt_lens`` valid tokens per row.
        Returns the flow velocity (noise - x0), (B, 64, H, W)."""
        B, _, H, W = x.shape
        dtype = x.dtype
        txt_len = context.shape[1]
        txt_lens = [txt_len] * B if txt_lens is None else [int(n) for n in txt_lens]

        hidden_states = torch.cat([self.txt_in(context), self.img_in(x.flatten(2).transpose(1, 2))], dim=1)
        pe = self.rope_freqs(txt_lens, txt_len, H, W, x.device)
        key_mask = None
        if any(n != txt_len for n in txt_lens):
            key_mask = torch.ones(B, hidden_states.shape[1], dtype=torch.bool, device=x.device)
            for i, n in enumerate(txt_lens):
                key_mask[i, n:txt_len] = False
        attn_fn = partial(text_image_attention, txt_len=txt_len, key_mask=key_mask)

        # the pipeline rounds t*1000 and t to the compute dtype; text tokens modulate from t = 0
        t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
        mod = (_split_rows(scale1), _split_rows(gate1.tanh()), _split_rows(scale2), _split_rows(gate2.tanh()))

        for block in self.transformer_blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                hidden_states = checkpoint(block, hidden_states, mod, pe, attn_fn, txt_len, use_reentrant=False)
            else:
                hidden_states = block(hidden_states, mod, pe, attn_fn, txt_len)

        hidden_states = self.proj_out(self.norm_out(hidden_states[:, txt_len:], temb[:-1]))
        return hidden_states.transpose(1, 2).reshape(B, self.out_channels, H, W)
