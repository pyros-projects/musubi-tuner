"""Qwen-Image 2.1 VAE (AutoencoderKLQwenImage21), single-image path only.

Port of ComfyUI's ``comfy/ldm/wan/vae2_2.py`` WanVAE as Comfy configures it for the
Qwen-Image 2.1 file: Wan 2.2 residual layout, temporal kernel 1, no patchify, RGBA in/out,
64 latent channels, 16x spatial compression. Images run with T = 1 and no temporal cache,
where every temporal conv reduces to its 2-D kernel and ``time_conv`` never runs (it stays
as a module only so the checkpoint loads strictly). Module names match the checkpoint.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from musubi_tuner.utils.safetensors_utils import load_safetensors

logger = logging.getLogger(__name__)

LATENT_CHANNELS = 64
SPATIAL_COMPRESSION = 16

# per-channel latent statistics from the diffusers/Comfy config; latents are cached as (raw - mean) / std
LATENTS_MEAN = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
    1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
    2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
    -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
    -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376,
    -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729,
    -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
    0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
]  # fmt: skip
LATENTS_STD = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910,
    3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756,
    3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467,
    3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125,
    3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959,
    3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161,
]  # fmt: skip


class CausalConv3d(nn.Conv3d):
    """Temporal kernel is 1 in this VAE, so on single frames this is a plain spatial conv."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim: int, images: bool = True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones((dim, 1, 1) if images else (dim, 1, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma.to(x)


def conv3x3(in_dim: int, out_dim: int) -> CausalConv3d:
    return CausalConv3d(in_dim, out_dim, (1, 3, 3), padding=(0, 1, 1))


class Resample(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        self.mode = mode
        if mode in ("upsample2d", "upsample3d"):
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim, 3, padding=1)
            )
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(dim, dim * 2, (1, 1, 1))
        elif mode in ("downsample2d", "downsample3d"):
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(dim, dim, (1, 1, 1), stride=(2, 1, 1))
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the first (only) frame never goes through time_conv
        b, c, t, h, w = x.shape
        x = self.resample(x.transpose(1, 2).reshape(b * t, c, h, w))
        return x.reshape(b, t, *x.shape[1:]).transpose(1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            conv3x3(in_dim, out_dim),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(0.0),
            conv3x3(out_dim, out_dim),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.shape
        x = self.norm(x.transpose(1, 2).reshape(b * t, c, h, w))
        q, k, v = (y.reshape(b * t, 1, c, h * w).transpose(2, 3) for y in self.to_qkv(x).chunk(3, dim=1))
        x = F.scaled_dot_product_attention(q, k, v).transpose(2, 3).reshape(b * t, c, h, w)
        x = self.proj(x)
        return x.reshape(b, t, c, h, w).transpose(1, 2) + identity


class AvgDown3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        B, C, T, H, W = x.shape
        ft, fs = self.factor_t, self.factor_s
        x = x.view(B, C, T // ft, ft, H // fs, fs, W // fs, fs)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, T // ft, H // fs, W // fs)
        x = x.view(B, self.out_channels, self.group_size, T // ft, H // fs, W // fs)
        return x.mean(dim=2)


class DupUp3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        ft, fs = self.factor_t, self.factor_s
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(x.size(0), self.out_channels, ft, fs, fs, x.size(2), x.size(3), x.size(4))
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(x.size(0), self.out_channels, x.size(2) * ft, x.size(4) * fs, x.size(6) * fs)
        if first_chunk:
            x = x[:, :, ft - 1 :, :, :]
        return x


class Down_ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, mult: int, temperal_downsample: bool, down_flag: bool):
        super().__init__()
        self.avg_shortcut = AvgDown3D(in_dim, out_dim, factor_t=2 if temperal_downsample else 1, factor_s=2 if down_flag else 1)
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim))
            in_dim = out_dim
        if down_flag:
            downsamples.append(Resample(out_dim, mode="downsample3d" if temperal_downsample else "downsample2d"))
        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsamples(x) + self.avg_shortcut(x)


class Up_ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, mult: int, temperal_upsample: bool, up_flag: bool):
        super().__init__()
        self.avg_shortcut = None
        if up_flag:
            self.avg_shortcut = DupUp3D(in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2)
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim))
            in_dim = out_dim
        if up_flag:
            upsamples.append(Resample(out_dim, mode="upsample3d" if temperal_upsample else "upsample2d"))
        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_main = self.upsamples(x)
        if self.avg_shortcut is not None:
            return x_main + self.avg_shortcut(x, first_chunk=True)
        return x_main


class Encoder3d(nn.Module):
    def __init__(self, dim, z_dim, dim_mult, num_res_blocks, temperal_downsample, in_channels):
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]
        self.conv1 = conv3x3(in_channels, dims[0])
        self.downsamples = nn.Sequential(
            *[
                Down_ResidualBlock(
                    in_dim,
                    out_dim,
                    num_res_blocks,
                    temperal_downsample=temperal_downsample[i] if i < len(temperal_downsample) else False,
                    down_flag=i != len(dim_mult) - 1,
                )
                for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]))
            ]
        )
        out_dim = dims[-1]
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim))
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), conv3x3(out_dim, z_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.middle(self.downsamples(self.conv1(x))))


class Decoder3d(nn.Module):
    def __init__(self, dim, z_dim, dim_mult, num_res_blocks, temperal_upsample, out_channels):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv1 = conv3x3(z_dim, dims[0])
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0]), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0]))
        self.upsamples = nn.Sequential(
            *[
                Up_ResidualBlock(
                    in_dim,
                    out_dim,
                    num_res_blocks + 1,
                    temperal_upsample=temperal_upsample[i] if i < len(temperal_upsample) else False,
                    up_flag=i != len(dim_mult) - 1,
                )
                for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]))
            ]
        )
        out_dim = dims[-1]
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), conv3x3(out_dim, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.upsamples(self.middle(self.conv1(x))))


class QwenImage21VAE(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        dec_dim: int = 144,
        z_dim: int = LATENT_CHANNELS,
        dim_mult=(1, 2, 4, 8, 8),
        num_res_blocks: int = 2,
        temperal_downsample=(False, True, True, True),
        image_channels: int = 4,
    ):
        super().__init__()
        dim_mult, temperal_downsample = list(dim_mult), list(temperal_downsample)
        self.z_dim = z_dim
        self.image_channels = image_channels
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, temperal_downsample, image_channels)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dec_dim, z_dim, dim_mult, num_res_blocks, temperal_downsample[::-1], image_channels)
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN).view(1, z_dim, 1, 1), persistent=False)
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD).view(1, z_dim, 1, 1), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.conv1.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) pixels in [-1, 1] -> (B, 64, H/16, W/16) raw latent mean. RGB gets opaque alpha."""
        if x.shape[1] == 3 and self.image_channels == 4:
            x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        return self.conv1(self.encoder(x.unsqueeze(2))).chunk(2, dim=1)[0].squeeze(2)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, 64, h, w) raw latent -> (B, 4, 16h, 16w) RGBA in about [-1, 1]."""
        return self.decoder(self.conv2(z.unsqueeze(2))).squeeze(2)

    def normalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.latents_mean.to(z)) / self.latents_std.to(z)

    def denormalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.latents_std.to(z) + self.latents_mean.to(z)

    @torch.no_grad()
    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, 3|4, H, W) in [-1, 1] -> normalized latents (B, 64, H/16, W/16), the training space."""
        return self.normalize_latents(self.encode(pixels.to(self.device, self.dtype)))

    @torch.no_grad()
    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized latents (B, 64, h, w) -> RGB (B, 3, 16h, 16w) in [0, 1]."""
        pixels = self.decode(self.denormalize_latents(latents.to(self.device, self.dtype)))
        return ((pixels[:, :3].float() + 1.0) / 2.0).clamp(0.0, 1.0)


def load_qwen_image21_vae(path: str, device="cpu", dtype: torch.dtype = torch.bfloat16) -> QwenImage21VAE:
    sd = load_safetensors(path, device=str(device), disable_mmap=True, dtype=dtype)
    with torch.device("meta"):
        vae = QwenImage21VAE(image_channels=sd["decoder.head.2.weight"].shape[0])
    vae.load_state_dict(sd, strict=True, assign=True)
    vae.latents_mean = torch.tensor(LATENTS_MEAN, device=device).view(1, -1, 1, 1)
    vae.latents_std = torch.tensor(LATENTS_STD, device=device).view(1, -1, 1, 1)
    logger.info(f"Loaded Qwen-Image 2.1 VAE from {path}")
    return vae.eval().requires_grad_(False)
