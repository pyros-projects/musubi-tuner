import sys
import types
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musubi_tuner import qwen_image_train_network
from musubi_tuner.qwen_image_train_network import QwenImageNetworkTrainer


class _FakeAccelerator:
    device = torch.device("cpu")


class _FakeQwenTransformer:
    in_channels = 16

    def __init__(self, events):
        self.events = events

    def to(self, device):
        self.events.append(f"transformer:to:{torch.device(device).type}")
        return self

    def __call__(self, **kwargs):
        self.events.append("transformer:denoise")
        return torch.zeros_like(kwargs["hidden_states"])


class _FakeVAE:
    dtype = torch.bfloat16

    def __init__(self, events):
        self.events = events

    def to(self, device):
        self.events.append(f"vae:to:{torch.device(device).type}")
        return self

    def eval(self):
        self.events.append("vae:eval")
        return self

    def decode_to_pixels(self, latents):
        self.events.append("vae:decode")
        return torch.zeros(latents.shape[0], 3, latents.shape[-2], latents.shape[-1])


class _EncodeFailsVAE(_FakeVAE):
    def encode_pixels_to_latents(self, tensor):
        raise AssertionError("sampling should use cached Qwen control latents")


class _FakeScheduler:
    order = 1

    def set_begin_index(self, index):
        pass

    def step(self, noise_pred, timestep, latents, return_dict=False):
        return (latents,)


def test_qwen_sample_with_offloading_moves_transformer_before_vae_decode():
    events = []
    trainer = QwenImageNetworkTrainer()
    trainer.is_edit = False
    trainer.is_layered = False
    args = types.SimpleNamespace(is_layered=False, sample_with_offloading=True)
    sample_parameter = {
        "vl_embed": torch.zeros(1, 1, 4),
        "negative_vl_embed": torch.zeros(1, 1, 4),
    }

    with (
        mock.patch.object(
            qwen_image_train_network.qwen_image_utils,
            "prepare_latents",
            return_value=torch.zeros(1, 1, 4, dtype=torch.bfloat16),
        ),
        mock.patch.object(qwen_image_train_network.qwen_image_utils, "calculate_shift_qwen_image", return_value=1.0),
        mock.patch.object(qwen_image_train_network.qwen_image_utils, "get_scheduler", return_value=_FakeScheduler()),
        mock.patch.object(
            qwen_image_train_network.qwen_image_utils,
            "retrieve_timesteps",
            return_value=([torch.tensor(1.0)], 1),
        ),
        mock.patch.object(
            qwen_image_train_network.qwen_image_utils,
            "unpack_latents",
            return_value=torch.zeros(1, 4, 1, 2, 2, dtype=torch.bfloat16),
        ),
        mock.patch.object(
            qwen_image_train_network,
            "clean_memory_on_device",
            side_effect=lambda device: events.append(f"clean:{torch.device(device).type}"),
        ),
    ):
        trainer.do_inference(
            _FakeAccelerator(),
            args,
            sample_parameter,
            _FakeVAE(events),
            torch.bfloat16,
            _FakeQwenTransformer(events),
            discrete_flow_shift=2.2,
            sample_steps=1,
            width=16,
            height=16,
            frame_count=1,
            generator=torch.Generator(device="cpu"),
            do_classifier_free_guidance=False,
            guidance_scale=1.0,
            cfg_scale=1.0,
        )

    assert events == [
        "transformer:denoise",
        "transformer:to:cpu",
        "clean:cpu",
        "vae:to:cpu",
        "vae:eval",
        "vae:decode",
        "vae:to:cpu",
        "clean:cpu",
    ]


def test_qwen_edit_sampling_uses_cached_control_latents_without_vae_encode():
    events = []
    trainer = QwenImageNetworkTrainer()
    trainer.is_edit = True
    trainer.is_layered = False
    args = types.SimpleNamespace(is_layered=False, sample_with_offloading=False)
    sample_parameter = {
        "vl_embed": torch.zeros(1, 1, 4),
        "negative_vl_embed": torch.zeros(1, 1, 4),
        "control_image_path": ["/tmp/ref.jpg"],
        "control_image_tensors": [torch.zeros(1, 3, 16, 16)],
        "qwen_control_latents": [torch.ones(1, 16, 1, 2, 2, dtype=torch.bfloat16)],
    }

    with (
        mock.patch.object(
            qwen_image_train_network.qwen_image_utils,
            "prepare_latents",
            return_value=torch.zeros(1, 1, 64, dtype=torch.bfloat16),
        ),
        mock.patch.object(qwen_image_train_network.qwen_image_utils, "calculate_shift_qwen_image", return_value=1.0),
        mock.patch.object(qwen_image_train_network.qwen_image_utils, "get_scheduler", return_value=_FakeScheduler()),
        mock.patch.object(
            qwen_image_train_network.qwen_image_utils,
            "retrieve_timesteps",
            return_value=([torch.tensor(1.0)], 1),
        ),
        mock.patch.object(
            qwen_image_train_network.qwen_image_utils,
            "unpack_latents",
            return_value=torch.zeros(1, 4, 1, 2, 2, dtype=torch.bfloat16),
        ),
        mock.patch.object(
            qwen_image_train_network,
            "clean_memory_on_device",
            side_effect=lambda device: events.append(f"clean:{torch.device(device).type}"),
        ),
    ):
        trainer.do_inference(
            _FakeAccelerator(),
            args,
            sample_parameter,
            _EncodeFailsVAE(events),
            torch.bfloat16,
            _FakeQwenTransformer(events),
            discrete_flow_shift=2.2,
            sample_steps=1,
            width=16,
            height=16,
            frame_count=1,
            generator=torch.Generator(device="cpu"),
            do_classifier_free_guidance=False,
            guidance_scale=1.0,
            cfg_scale=1.0,
        )

    assert "vae:decode" in events
