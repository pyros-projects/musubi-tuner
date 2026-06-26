import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musubi_tuner import krea2_generate_image
from musubi_tuner.krea2 import krea2_sampling


class _FakeTransformer:
    def __init__(self, events: list[str], blocks_to_swap: int = 0):
        self.events = events
        self.blocks_to_swap = blocks_to_swap
        self.config = types.SimpleNamespace(patch=1, channels=3)

    def _wait_for_pending_block_swaps(self):
        self.events.append("transformer:wait")

    def move_to_device_except_swap_blocks(self, device):
        self.events.append(f"transformer:move_except:{torch.device(device).type}")
        return self

    def prepare_block_swap_before_forward(self):
        self.events.append("transformer:prepare")

    def to(self, device):
        self.events.append(f"transformer:to:{torch.device(device).type}")
        return self

    def __call__(self, *, img, context, t, pos, mask):
        self.events.append("transformer:denoise")
        return torch.zeros_like(img)


class _FakeVAE:
    temperal_downsample = []
    z_dim = 3
    dtype = torch.bfloat16

    def __init__(self, events: list[str], fail_decode: bool = False):
        self.events = events
        self.fail_decode = fail_decode

    def to(self, device):
        self.events.append(f"vae:to:{torch.device(device).type}")
        return self

    def eval(self):
        self.events.append("vae:eval")
        return self

    def decode_to_pixels(self, latent):
        self.events.append("vae:decode")
        if self.fail_decode:
            raise RuntimeError("decode failed")
        return torch.zeros(latent.shape[0], 3, latent.shape[-2], latent.shape[-1], dtype=torch.float32)


def _record_clean(events: list[str]):
    def clean(device):
        events.append(f"clean:{torch.device(device).type}")

    return clean


def test_decode_offload_context_moves_transformer_before_vae_decode():
    events: list[str] = []
    transformer = _FakeTransformer(events, blocks_to_swap=2)
    vae = _FakeVAE(events)

    with krea2_sampling.transformer_decode_offload(
        transformer,
        torch.device("cuda"),
        enabled=True,
        restore_after=False,
        clean_fn=_record_clean(events),
    ):
        vae.to(torch.device("cuda"))
        vae.eval()
        vae.decode_to_pixels(torch.zeros(1, 3, 1, 2, 2))

    assert events == [
        "transformer:wait",
        "transformer:to:cpu",
        "clean:cuda",
        "vae:to:cuda",
        "vae:eval",
        "vae:decode",
    ]


def test_decode_offload_context_can_restore_transformer_after_success():
    events: list[str] = []
    transformer = _FakeTransformer(events, blocks_to_swap=2)

    with krea2_sampling.transformer_decode_offload(
        transformer,
        torch.device("cuda"),
        enabled=True,
        restore_after=True,
        clean_fn=_record_clean(events),
    ):
        events.append("decode")

    assert events == [
        "transformer:wait",
        "transformer:to:cpu",
        "clean:cuda",
        "decode",
        "transformer:move_except:cuda",
        "transformer:prepare",
        "clean:cuda",
    ]


def test_decode_offload_context_restores_transformer_after_decode_failure():
    events: list[str] = []
    transformer = _FakeTransformer(events, blocks_to_swap=2)

    with pytest.raises(RuntimeError, match="decode failed"):
        with krea2_sampling.transformer_decode_offload(
            transformer,
            torch.device("cuda"),
            enabled=True,
            restore_after=True,
            clean_fn=_record_clean(events),
        ):
            events.append("decode")
            raise RuntimeError("decode failed")

    assert events == [
        "transformer:wait",
        "transformer:to:cpu",
        "clean:cuda",
        "decode",
        "transformer:move_except:cuda",
        "transformer:prepare",
        "clean:cuda",
    ]


def test_decode_offload_context_is_noop_when_disabled():
    events: list[str] = []
    transformer = _FakeTransformer(events, blocks_to_swap=0)

    with krea2_sampling.transformer_decode_offload(
        transformer,
        torch.device("cuda"),
        enabled=False,
        restore_after=True,
        clean_fn=_record_clean(events),
    ):
        events.append("decode")

    assert events == ["decode"]


def test_parse_args_accepts_sample_with_offloading_flag():
    with mock.patch.object(
        sys,
        "argv",
        [
            "krea2_generate_image.py",
            "hello",
            "--dit",
            "/tmp/dit.safetensors",
            "--vae",
            "/tmp/vae.safetensors",
            "--text_encoder",
            "/tmp/text_encoder.safetensors",
            "--save_path",
            "/tmp/output",
            "--sample_with_offloading",
        ],
    ):
        args = krea2_generate_image.parse_args()

    assert args.sample_with_offloading is True


def test_generate_passes_decode_offload_to_sampler():
    args = types.SimpleNamespace(
        seed=123,
        prompt="hello",
        num_images=1,
        negative_prompt="",
        guidance_scale=1.0,
        blocks_to_swap=0,
        width=2,
        height=2,
        steps=1,
        y1=0.5,
        y2=1.15,
        mu=None,
        sample_with_offloading=True,
    )
    embeds = (
        torch.zeros(1, 1, 1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        None,
        None,
    )

    with (
        mock.patch.object(krea2_generate_image, "encode", return_value=embeds),
        mock.patch.object(krea2_generate_image, "sample", return_value=[]) as sample_mock,
    ):
        krea2_generate_image.generate(args, object(), object(), object(), "cuda", torch.bfloat16, "cuda")

    assert sample_mock.call_args.kwargs["offload_transformer_for_decode"] is True
    assert sample_mock.call_args.kwargs["restore_transformer_after_decode"] is True


def test_standalone_sample_offloads_transformer_before_decode_and_restores():
    events: list[str] = []
    transformer = _FakeTransformer(events, blocks_to_swap=2)
    vae = _FakeVAE(events)
    txt = torch.zeros(1, 1, 1, 1)
    txtmask = torch.ones(1, 1, dtype=torch.bool)

    images = krea2_sampling.sample(
        transformer,
        vae,
        txt,
        txtmask,
        device="cpu",
        dtype=torch.bfloat16,
        width=2,
        height=2,
        steps=1,
        cfg_scale=1.0,
        offload_transformer_for_decode=True,
        restore_transformer_after_decode=True,
        clean_fn=_record_clean(events),
    )

    assert len(images) == 1
    assert events == [
        "transformer:denoise",
        "transformer:wait",
        "transformer:to:cpu",
        "clean:cpu",
        "vae:to:cpu",
        "vae:decode",
        "vae:to:cpu",
        "transformer:to:cpu",
        "clean:cpu",
    ]
