"""MiniMax H3 video VAE components used by caching and training previews."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from musubi_tuner.minimax_h3.minimax_h3_utils import load_selected_weights, resolve_safetensor_files

# Decoder architecture follows the Apache-2.0 MiniMax-H3 implementation from
# Hugging Face Diffusers PR #14355 (commit abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc).

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LATENTS_MEAN = (
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
)
LATENTS_STD = (
    1.2223774194717407,
    1.2767263650894165,
    1.68317747116088865,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.96531379222869875,
    1.05698859691619875,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049925,
    0.7197399735450745,
    0.69362932443618775,
    2.961095094680786,
    2.7694199085235595,
    3.0496184825897215,
    2.1088054180145265,
    3.276226282119751,
    3.1627357006073,
    2.28168129920959475,
    2.6127843856811525,
)


class CausalConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)

    def forward(self, x):
        if sum(self.causal_padding) == 0:
            return super().forward(x)
        pt, ph, pw = self.causal_padding
        x = F.pad(x, (pw, pw, ph, ph, 0, 0), mode="reflect")
        if x.shape[2] == 1:
            # The preceding causal taps multiply zero padding. Avoid materializing them.
            return F.conv3d(x, self.weight[:, :, -1:], self.bias, self.stride, 0, self.dilation, self.groups)
        x = F.pad(x, (0, 0, 0, 0, pt * 2, 0))
        return super().forward(x)


class TemporalIsolatedGroupNorm(nn.GroupNorm):
    def forward(self, x):
        if x.ndim != 5:
            return super().forward(x)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, 1, h, w)
        return super().forward(x).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def group_norm_3d(channels):
    return TemporalIsolatedGroupNorm(32, channels, eps=1e-6, affine=True)


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2):
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(in_channels, out_channels, 3, padding=(1, 0, 0), stride=(time_stride, space_stride, space_stride))

    def forward(self, x):
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = group_norm_3d(in_channels)
        self.norm2 = group_norm_3d(self.out_channels)
        self.conv1 = CausalConv3d(in_channels, self.out_channels, 3, padding=1)
        self.conv2 = CausalConv3d(self.out_channels, self.out_channels, 3, padding=1)
        if in_channels != self.out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, self.out_channels, 1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return h + x


class EncoderFCN3D(nn.Module):
    def __init__(
        self,
        ch=128,
        ch_mult=(1, 2, 2, 4, 4, 8),
        space_down=(2, 2, 2, 2, 1, 1),
        time_down=(1, 2, 2, 1, 1, 1),
        num_res_blocks=2,
        in_channels=3,
        z_channels=24,
    ):
        super().__init__()
        levels = len(ch_mult)
        counts = [num_res_blocks] * levels if isinstance(num_res_blocks, int) else list(num_res_blocks)
        block_mid = [ch * value for value in ch_mult]
        block_in = [block_mid[0], *block_mid[:-1]]
        self.num_res_blocks = counts
        self.conv_in = CausalConv3d(in_channels, block_in[0], 3, padding=1)
        self.down = nn.ModuleList()
        for level in range(levels):
            down = nn.Module()
            down.block = nn.ModuleList(
                [
                    ResnetBlock3D(block_in[level] if index == 0 else block_mid[level], block_mid[level])
                    for index in range(counts[level])
                ]
            )
            if space_down[level] * time_down[level] > 1:
                down.downsample = Downsample3D(
                    block_mid[level], block_mid[level], time_stride=time_down[level], space_stride=space_down[level]
                )
            self.down.append(down)
        self.norm_out = group_norm_3d(block_mid[-1])
        self.conv_out = CausalConv3d(block_mid[-1], 2 * z_channels, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for level, down in enumerate(self.down):
            for block in down.block:
                h = block(h)
            if hasattr(down, "downsample"):
                h = down.downsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class MiniMaxH3VideoEncoder(nn.Module):
    def __init__(self, tile_size=256, tile_overlap=64, tiling=True):
        super().__init__()
        self.vae_ratio = 16
        self.clip_length = 17
        self.token_drop = 3
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap
        self.tiling = tiling
        self.encoder = EncoderFCN3D()
        self.quant_conv = nn.Conv3d(48, 48, 1)
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN), persistent=False)
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD), persistent=False)
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    @property
    def device(self):
        return self.quant_conv.weight.device

    @property
    def dtype(self):
        return self.quant_conv.weight.dtype

    def _encode_moments(self, x):
        return self.quant_conv(self.encoder(x))

    def split_tiles(self, input_len):
        if self.tile_size >= input_len:
            return [0], [input_len], []
        count = math.ceil(input_len / self.tile_size)
        while True:
            overlaps = [self.tile_overlap_min] * (count - 1)
            remaining = self.tile_size * count - sum(overlaps) - input_len
            if remaining >= 0:
                break
            count += 1
        for index in range(remaining // self.vae_ratio):
            overlaps[index % (count - 1)] += self.vae_ratio
        starts = [0]
        for index in range(count - 1):
            starts.append(starts[-1] + self.tile_size - overlaps[index])
        return starts, [self.tile_size] * count, overlaps

    @staticmethod
    def blend(a, b, extent, dim):
        extent = min(a.shape[dim], b.shape[dim], extent)
        if extent == 0:
            return b
        pos = torch.arange(extent, device=b.device, dtype=b.dtype) / extent
        shape = [1] * b.ndim
        shape[dim] = extent
        pos = pos.view(shape)
        a_slice = [slice(None)] * a.ndim
        b_slice = [slice(None)] * b.ndim
        a_slice[dim] = slice(-extent, None)
        b_slice[dim] = slice(0, extent)
        blended = a[tuple(a_slice)] * (1 - pos) + b[tuple(b_slice)] * pos
        if extent == b.shape[dim]:
            return blended
        b_slice[dim] = slice(extent, None)
        return torch.cat([blended, b[tuple(b_slice)]], dim=dim)

    def tiled_encode(self, x):
        y_starts, y_lengths, y_overlaps = self.split_tiles(x.shape[-2])
        x_starts, x_lengths, x_overlaps = self.split_tiles(x.shape[-1])
        rows = []
        for y, yl in zip(y_starts, y_lengths):
            rows.append([self._encode_moments(x[..., y : y + yl, xx : xx + xl]) for xx, xl in zip(x_starts, x_lengths)])
        ly = [value // self.vae_ratio for value in y_overlaps]
        lx = [value // self.vae_ratio for value in x_overlaps]
        output_rows = []
        for i, row in enumerate(rows):
            output = []
            for j, tile in enumerate(row):
                if i:
                    tile = self.blend(rows[i - 1][j], tile, ly[i - 1], -2)
                if j:
                    tile = self.blend(row[j - 1], tile, lx[j - 1], -1)
                if i < len(rows) - 1:
                    tile = tile[..., : -ly[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -lx[j]]
                output.append(tile)
            output_rows.append(torch.cat(output, dim=-1))
        return torch.cat(output_rows, dim=-2)

    def _adaptive_encode(self, x):
        return self.tiled_encode(x) if self.tiling else self._encode_moments(x)

    def encode(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        x = ((x + 1.0) * 0.5 - self.pixel_mean.to(x)) / self.pixel_std.to(x)
        if x.shape[2] == 1:
            moments = self._adaptive_encode(x)[:, :, -1:]
        else:
            pad = (-x.shape[2]) % self.clip_length
            if pad:
                x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)
            moments = torch.cat(
                [self._adaptive_encode(x[:, :, i : i + self.clip_length]) for i in range(0, x.shape[2], self.clip_length)],
                dim=2,
            )[:, :, : -self.token_drop]
        mean = moments.float().chunk(2, dim=1)[0]
        mean_value = self.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        std_value = self.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return (mean - mean_value) / std_value


def _create_token_ids(patch_dims, device, dtype=torch.float32):
    grids = [2.0 * (torch.arange(0.5, size, dtype=dtype, device=device) / size) - 1.0 for size in patch_dims]
    return torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, len(patch_dims) - 1).unsqueeze(0)


class RotaryEmbeddingND(nn.Module):
    def __init__(self, dim, rotary_base=100.0, n_dim=3):
        super().__init__()
        if dim % (2 * n_dim):
            raise ValueError(f"Rotary dimension {dim} must be divisible by {2 * n_dim}")
        self.dim = dim
        self.rotary_base = rotary_base
        self.n_dim = n_dim
        self.register_buffer("inv_freq", self._make_inv_freq(), persistent=False)

    def _make_inv_freq(self, device=None):
        return 1.0 / self.rotary_base ** torch.arange(0, 1, 2 * self.n_dim / self.dim, dtype=torch.float32, device=device)

    def materialize(self, device):
        self.inv_freq = self._make_inv_freq(device)

    def forward(self, position_ids):
        angles = 2.0 * math.pi * position_ids[:, :, :, None].float() * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


def _apply_rotary_emb(hidden_states, rotary_emb):
    cosine, sine = (value.to(hidden_states.dtype) for value in rotary_emb)
    rotary_dim = cosine.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cosine + rotated * sine, passthrough), dim=-1)


class DecoderFeedForward(nn.Module):
    def __init__(self, dim, mult=4, bias=True):
        super().__init__()
        inner_dim = dim * mult
        self.w1 = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.w2 = nn.Linear(inner_dim, dim, bias=bias)

    def forward(self, hidden_states):
        gate, value = self.w1(hidden_states).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * value)


class DecoderAttention(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = heads * dim_head
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_qkv = nn.Linear(inner_dim, inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(inner_dim, inner_dim, bias=bias)

    def forward(self, hidden_states, rotary_emb=None):
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = self.to_qkv(hidden_states).view(batch_size, sequence_length, self.heads, 3 * self.dim_head)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self.norm_q(query.float()).to(query.dtype)
        key = self.norm_k(key.float()).to(key.dtype)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, rotary_emb)
            key = _apply_rotary_emb(key, rotary_emb)
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        return self.to_out(hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, -1))


class DecoderTransformerBlock(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5):
        super().__init__()
        dim = heads * dim_head
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=True, eps=eps)
        self.attn = DecoderAttention(heads, dim_head, bias, eps)
        self.scale1 = nn.Parameter(torch.empty(dim))
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=True, eps=eps)
        self.ff = DecoderFeedForward(dim, bias=bias)
        self.scale2 = nn.Parameter(torch.empty(dim))

    def forward(self, hidden_states, rotary_emb=None):
        normalized = F.rms_norm(
            hidden_states.float(),
            (hidden_states.shape[-1],),
            self.norm1.weight.float(),
            self.norm1.eps,
        ).to(hidden_states.dtype)
        hidden_states = hidden_states + self.attn(normalized, rotary_emb) * self.scale1
        normalized = F.rms_norm(
            hidden_states.float(),
            (hidden_states.shape[-1],),
            self.norm2.weight.float(),
            self.norm2.eps,
        ).to(hidden_states.dtype)
        return hidden_states + self.ff(normalized) * self.scale2


class ViT3DDecoder(nn.Module):
    def __init__(
        self,
        patch_size=16,
        patch_size_t=4,
        in_channels=24,
        out_channels=3,
        num_layers=36,
        heads=32,
        dim_head=64,
        rope_theta=100.0,
        rope_dim_ratio=0.75,
        bias=True,
        eps=1e-5,
        num_register_tokens=4,
    ):
        super().__init__()
        dim = heads * dim_head
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.pos_embed = RotaryEmbeddingND(int(dim_head * rope_dim_ratio), rope_theta, n_dim=3)
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, dim))
        # Unused at inference; retained so the released checkpoint loads exactly.
        self.register_buffer("mask_token", torch.empty(1, 1, dim))
        self.transformer_blocks = nn.ModuleList([DecoderTransformerBlock(heads, dim_head, bias, eps) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def forward(self, hidden_states):
        batch_size, channels, frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(batch_size, frames * height * width, channels)
        hidden_states = self.x_embedder(hidden_states)
        num_patches = hidden_states.shape[1]
        hidden_states = torch.cat(
            (
                hidden_states,
                self.register_tokens.expand(batch_size, -1, -1),
                torch.zeros_like(hidden_states[:, :1]),
            ),
            dim=1,
        )
        position_ids = _create_token_ids((frames, height, width), hidden_states.device).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
        rotary_emb = self.pos_embed(torch.cat((position_ids, suffix_ids), dim=1))
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)
        hidden_states = self.proj_out(self.norm_out(hidden_states))[:, :num_patches]
        hidden_states = hidden_states.view(
            batch_size,
            frames,
            height,
            width,
            self.out_channels,
            self.patch_size_t,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            frames * self.patch_size_t,
            height * self.patch_size,
            width * self.patch_size,
        )


def decode_single_frame_latent(latents: torch.Tensor, decode_video: Callable[[torch.Tensor], torch.Tensor]):
    """Decode a T=1 latent through the decoder's minimum in-distribution T=2 shape.

    The final raw frame reconstructs the still best; roundtrip PSNR on real
    images measured it 8-9 dB above both the lone-T=1 decode and the
    duplicated clip's earlier frame positions.
    """

    if latents.shape[2] != 1:
        raise ValueError("decode_single_frame_latent expects exactly one temporal latent")
    return decode_video(torch.cat((latents, latents), dim=2))[:, :, -1:]


def _select_sharpest_frame(frames: torch.Tensor) -> torch.Tensor:
    """Pick the temporal frame with the highest Laplacian variance."""

    batch, _, length, height, width = frames.shape
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=frames.device, dtype=torch.float32)
    gray = (frames.float() * weights.view(1, 3, 1, 1, 1)).sum(dim=1)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=frames.device, dtype=torch.float32
    ).view(1, 1, 3, 3)
    laplacian = F.conv2d(gray.reshape(batch * length, 1, height, width), kernel, padding=1)
    index = int(laplacian.var(dim=(1, 2, 3)).view(batch, length).mean(dim=0).argmax())
    return frames[:, :, index : index + 1]


# Chunked temporal decode constants, ported from the Apache-2.0 ai-toolkit
# MiniMax H3 implementation (minimax_h3/src/vae.py). A 17-frame clip maps to 5
# latents; the decoder emits 4 raw frames per latent, the first 3 of a chunk
# are temporal pre-padding, chunks overlap by 2 latents and cross-fade over 5
# pixel frames.
_CLIP_LENGTH = 17
_TEMPORAL_RATIO = 4
_TOKEN_DROP = 3
_FRAME_PRE_PADDING = (-_CLIP_LENGTH) % _TEMPORAL_RATIO
_TOKENS_CHUNK_SIZE = math.ceil(_CLIP_LENGTH / _TEMPORAL_RATIO)
_TOKEN_OVERLAP = (-_TOKEN_DROP) % _TOKENS_CHUNK_SIZE
_FRAME_OVERLAP = max(_TOKEN_OVERLAP * _TEMPORAL_RATIO - _FRAME_PRE_PADDING, 0)


def _blend_time(first: torch.Tensor, second: torch.Tensor, extent: int) -> torch.Tensor:
    extent = min(first.shape[2], second.shape[2], extent)
    if extent == 0:
        return second
    weights = torch.arange(extent, device=second.device, dtype=second.dtype) / extent
    weights = weights.view(1, 1, extent, 1, 1)
    blended = first[:, :, -extent:] * (1.0 - weights) + second[:, :, :extent] * weights
    if extent == second.shape[2]:
        return blended
    return torch.cat((blended, second[:, :, extent:]), dim=2)


def decode_video_latents(latents: torch.Tensor, decode_clip: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Chunked H3 video decode: ``5n+2`` latents to ``17n+5`` pixel frames.

    Faithful port of the ai-toolkit reference decode, including its handling
    of clips shorter than one chunk (padded by repeating the final latent).
    """

    tcs = _TOKENS_CHUNK_SIZE
    chunk_frames = tcs * _TEMPORAL_RATIO
    split_count = 2 if _TOKEN_DROP > 0 else 1

    num_tokens = latents.shape[2] + _TOKEN_DROP
    pad_tokens = (-num_tokens) % tcs
    num_chunks = (num_tokens + pad_tokens) // tcs - (split_count - 1)
    if num_chunks < 1:
        pad_tokens += tcs
        num_chunks += 1
    if pad_tokens > 0:
        latents = torch.cat([latents, latents[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    decoded = []
    overlap = None
    for index in range(num_chunks):
        start = index * tcs
        clip = decode_clip(latents[:, :, start : start + tcs + _TOKEN_OVERLAP])
        for part_index in range(split_count):
            part = clip[:, :, part_index * chunk_frames : (part_index + 1) * chunk_frames]
            part = part[:, :, _FRAME_PRE_PADDING :]
            if part_index == 0:
                if overlap is not None:
                    part = _blend_time(overlap, part, _FRAME_OVERLAP)
                decoded.append(part)
            else:
                overlap = part
    if overlap is not None:
        decoded.append(overlap)
    frames = torch.cat(decoded, dim=2)

    if pad_tokens > 0:
        # Repeated latents produced trailing frames that were never requested;
        # a chunk's final token covers clip_length % ratio frames, others cover
        # the full temporal ratio.
        intra_tail = _CLIP_LENGTH % _TEMPORAL_RATIO
        before_pad = latents.shape[2] - pad_tokens
        pad_frames = sum(
            intra_tail if intra_tail and (before_pad + offset) % tcs == 0 else _TEMPORAL_RATIO
            for offset in range(pad_tokens)
        )
        frames = frames[:, :, :-pad_frames]
    return frames


def select_preview_frame(frames: torch.Tensor, frame_select: str) -> torch.Tensor:
    """Reduce a decoded clip to the single preview frame."""

    if frame_select == "first":
        return frames[:, :, :1]
    if frame_select == "last":
        return frames[:, :, -1:]
    if frame_select == "sharpest":
        return _select_sharpest_frame(frames)
    raise ValueError(f"Unknown H3 preview frame_select: {frame_select}")


class MiniMaxH3VideoDecoder(nn.Module):
    def __init__(self, tile_size=256, tile_overlap=64, tiling=True):
        super().__init__()
        self.vae_ratio = 16
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap
        self.tiling = tiling
        self.post_quant_conv = nn.Conv3d(24, 24, 1)
        self.decoder = ViT3DDecoder()
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    @property
    def device(self):
        return self.post_quant_conv.weight.device

    @property
    def dtype(self):
        return self.post_quant_conv.weight.dtype

    def _split_tiles(self, length):
        if self.tile_size >= length:
            return [0], [length], []
        count = math.ceil(length / self.tile_size)
        while self.tile_size * count - self.tile_overlap_min * (count - 1) < length:
            count += 1
        overlaps = [self.tile_overlap_min] * (count - 1)
        remaining = self.tile_size * count - sum(overlaps) - length
        for index in range(remaining // self.vae_ratio):
            overlaps[index % (count - 1)] += self.vae_ratio
        starts = [0]
        for index in range(count - 1):
            starts.append(starts[-1] + self.tile_size - overlaps[index])
        return starts, [self.tile_size] * count, overlaps

    @staticmethod
    def _blend(first, second, extent, dim):
        extent = min(first.shape[dim], second.shape[dim], extent)
        positions = torch.arange(extent, device=second.device, dtype=second.dtype)
        shape = [1] * first.ndim
        shape[dim] = extent
        first_weight = (1 - positions / extent).view(shape)
        second_weight = (positions / extent).view(shape)
        first_slice = [slice(None)] * first.ndim
        first_slice[dim] = slice(-extent, None)
        second_slice = [slice(None)] * second.ndim
        second_slice[dim] = slice(0, extent)
        blended = first[tuple(first_slice)] * first_weight + second[tuple(second_slice)] * second_weight
        if extent == second.shape[dim]:
            return blended
        rest = [slice(None)] * second.ndim
        rest[dim] = slice(extent, None)
        return torch.cat((blended, second[tuple(rest)]), dim=dim)

    def _stitch_tiles(self, tiles, height_overlaps, width_overlaps):
        stitched_rows = []
        for row_index, row in enumerate(tiles):
            stitched_row = []
            for column_index, tile in enumerate(row):
                if row_index:
                    tile = self._blend(tiles[row_index - 1][column_index], tile, height_overlaps[row_index - 1], -2)
                if column_index:
                    tile = self._blend(row[column_index - 1], tile, width_overlaps[column_index - 1], -1)
                if row_index < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[row_index], :]
                if column_index < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[column_index]]
                stitched_row.append(tile)
            stitched_rows.append(torch.cat(stitched_row, dim=-1))
        return torch.cat(stitched_rows, dim=-2)

    def _decode_clip(self, latents):
        if not self.tiling:
            return self.decoder(self.post_quant_conv(latents))
        height_starts, height_lengths, height_overlaps = self._split_tiles(latents.shape[-2] * self.vae_ratio)
        width_starts, width_lengths, width_overlaps = self._split_tiles(latents.shape[-1] * self.vae_ratio)
        tiles = [
            [
                self.decoder(
                    self.post_quant_conv(
                        latents[
                            ...,
                            top // self.vae_ratio : (top + tile_height) // self.vae_ratio,
                            left // self.vae_ratio : (left + tile_width) // self.vae_ratio,
                        ]
                    )
                )
                for left, tile_width in zip(width_starts, width_lengths)
            ]
            for top, tile_height in zip(height_starts, height_lengths)
        ]
        return self._stitch_tiles(tiles, height_overlaps, width_overlaps)

    def _denormalize_latents(self, latents):
        latent_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latents)
        latent_std = self.latents_std.view(1, -1, 1, 1, 1).to(latents)
        return (latents.float() * latent_std + latent_mean).to(self.dtype)

    def _denormalize_pixels(self, pixels):
        pixels = pixels.float() * self.pixel_std.to(pixels) + self.pixel_mean.to(pixels)
        return pixels.clamp(0.0, 1.0) * 2.0 - 1.0

    def decode(self, latents, frame_select: str = "dup_last"):
        """Decode a one- or two-latent preview to a single image frame."""

        latents = self._denormalize_latents(latents)
        if latents.shape[2] == 1:
            pixels = decode_single_frame_latent(latents, self._decode_clip)
        elif latents.shape[2] == 2:
            if frame_select == "dup_last":
                # Keep the packet's second latent and decode it through the
                # measured-best duplicate-and-keep-last-frame path.
                pixels = decode_single_frame_latent(latents[:, :, 1:], self._decode_clip)
            else:
                # Proper chunked decode (pads to a full chunk internally), then
                # reduce the natural 5-frame clip to one image.
                frames = decode_video_latents(latents, self._decode_clip)
                pixels = select_preview_frame(frames, frame_select)
        else:
            raise ValueError("H3 image preview decoding supports one or two temporal latents")
        return self._denormalize_pixels(pixels)

    def decode_video(self, latents):
        """Decode a full ``5n+2``-latent video to all ``17n+5`` pixel frames."""

        latents = self._denormalize_latents(latents)
        frames = decode_video_latents(latents, self._decode_clip)
        return self._denormalize_pixels(frames)


def load_video_vae(
    path: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    tile_size: int = 256,
    tile_overlap: int = 64,
    tiling: bool = True,
    disable_numpy_memmap: bool = False,
) -> MiniMaxH3VideoEncoder:
    with torch.device("meta"):
        model = MiniMaxH3VideoEncoder(tile_size, tile_overlap, tiling)
    load_selected_weights(
        model,
        resolve_safetensor_files(path, "video_vae"),
        device=device,
        dtype=dtype,
        disable_numpy_memmap=disable_numpy_memmap,
    )
    # Non-persistent constants were created on meta with the module.
    model.latents_mean = torch.tensor(LATENTS_MEAN, device=device)
    model.latents_std = torch.tensor(LATENTS_STD, device=device)
    model.pixel_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1, 1)
    model.pixel_std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1, 1)
    model.eval().requires_grad_(False)
    return model


def load_video_vae_decoder(
    path: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float16,
    tile_size: int = 256,
    tile_overlap: int = 64,
    tiling: bool = True,
    disable_numpy_memmap: bool = False,
) -> MiniMaxH3VideoDecoder:
    """Load only the decoder half needed for scheduled training previews."""

    with torch.device("meta"):
        model = MiniMaxH3VideoDecoder(tile_size, tile_overlap, tiling)
    load_selected_weights(
        model,
        resolve_safetensor_files(path, "video_vae"),
        device=device,
        dtype=dtype,
        disable_numpy_memmap=disable_numpy_memmap,
    )
    model.pixel_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1, 1)
    model.pixel_std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1, 1)
    model.decoder.pos_embed.materialize(device)
    model.eval().requires_grad_(False)
    return model
