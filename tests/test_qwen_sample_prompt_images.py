import sys
import types
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _FakeCPUAccelerator:
    device = torch.device("cpu")


class _RecordingQwenVAE:
    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self):
        self.encode_inputs = []

    def to(self, *args, **kwargs):
        if "device" in kwargs and kwargs["device"] is not None:
            self.device = torch.device(kwargs["device"])
        elif args:
            try:
                self.device = torch.device(args[0])
            except (TypeError, RuntimeError):
                pass
        if "dtype" in kwargs and kwargs["dtype"] is not None:
            self.dtype = kwargs["dtype"]
        return self

    def eval(self):
        return self

    def encode_pixels_to_latents(self, tensor):
        self.encode_inputs.append(tensor.detach().cpu())
        value = float(len(self.encode_inputs))
        return torch.full((tensor.shape[0], 16, 1, 2, 2), value, dtype=torch.float32)


def _patch_qwen_prompt_dependencies(monkeypatch, prompts, seen_preprocess_paths, image_embed_calls, text_embed_calls):
    import musubi_tuner.qwen_image_train_network as qwen_train

    monkeypatch.setattr(qwen_train, "load_prompts", lambda _sample_prompts: prompts)
    monkeypatch.setattr(
        qwen_train.qwen_image_utils,
        "load_qwen2_5_vl",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(qwen_train.qwen_image_utils, "load_vl_processor", lambda: object())

    def fake_preprocess_control_image(control_image_path, *args, **kwargs):
        seen_preprocess_paths.append(control_image_path)
        return torch.zeros(1, 3, 8, 8), np.zeros((8, 8, 3), dtype=np.float32), None

    def fake_get_qwen_prompt_embeds(*args, **kwargs):
        text_embed_calls.append(args[2])
        return torch.ones(1, 3, 4), torch.ones(1, 3, dtype=torch.bool)

    def fake_get_qwen_prompt_embeds_with_image(*args, **kwargs):
        image_embed_calls.append((args[2], args[3]))
        return torch.ones(1, 3, 4), torch.ones(1, 3, dtype=torch.bool)

    monkeypatch.setattr(qwen_train.qwen_image_utils, "preprocess_control_image", fake_preprocess_control_image)
    monkeypatch.setattr(qwen_train.qwen_image_utils, "get_qwen_prompt_embeds", fake_get_qwen_prompt_embeds)
    monkeypatch.setattr(qwen_train.qwen_image_utils, "get_qwen_prompt_embeds_with_image", fake_get_qwen_prompt_embeds_with_image)


def _qwen_edit_args():
    return types.SimpleNamespace(
        fp8_vl=False,
        is_layered=False,
        model_version="edit-2511",
        text_encoder="/tmp/qwen-vl.safetensors",
        vae="/tmp/qwen-vae.safetensors",
        vae_dtype="float32",
    )


def _recording_trainer():
    from musubi_tuner.qwen_image_train_network import QwenImageNetworkTrainer

    class RecordingTrainer(QwenImageNetworkTrainer):
        def __init__(self):
            super().__init__()
            self.loaded_vaes = []

        def load_vae(self, args, vae_dtype, vae_path):
            vae = _RecordingQwenVAE()
            self.loaded_vaes.append((vae, vae_dtype, vae_path))
            return vae

    trainer = RecordingTrainer()
    trainer.is_edit = True
    trainer.is_layered = False
    return trainer


def test_qwen_edit_sample_normalizes_control_image_path_string(tmp_path, monkeypatch):
    image = tmp_path / "ref.jpg"
    image.write_bytes(b"image")
    prompts = [{"prompt": "make it cinematic", "control_image_path": str(image), "width": 64, "height": 64}]
    seen_preprocess_paths = []
    image_embed_calls = []
    text_embed_calls = []
    _patch_qwen_prompt_dependencies(monkeypatch, prompts, seen_preprocess_paths, image_embed_calls, text_embed_calls)

    trainer = _recording_trainer()

    sample_parameters = trainer.process_sample_prompts(_qwen_edit_args(), _FakeCPUAccelerator(), str(tmp_path / "prompts.toml"))

    assert seen_preprocess_paths == [str(image.resolve())]
    assert sample_parameters[0]["control_image_path"] == [str(image.resolve())]
    assert image_embed_calls
    assert text_embed_calls == []


def test_qwen_edit_sample_accepts_image_path_alias(tmp_path, monkeypatch):
    image = tmp_path / "ref.jpg"
    image.write_bytes(b"image")
    prompts = [{"prompt": "make it cinematic", "image_path": image.name, "width": 64, "height": 64}]
    seen_preprocess_paths = []
    image_embed_calls = []
    text_embed_calls = []
    _patch_qwen_prompt_dependencies(monkeypatch, prompts, seen_preprocess_paths, image_embed_calls, text_embed_calls)

    trainer = _recording_trainer()

    sample_parameters = trainer.process_sample_prompts(_qwen_edit_args(), _FakeCPUAccelerator(), str(tmp_path / "prompts.toml"))

    assert seen_preprocess_paths == [str(image.resolve())]
    assert sample_parameters[0]["control_image_path"] == [str(image.resolve())]
    assert image_embed_calls
    assert text_embed_calls == []


def test_qwen_edit_sample_precaches_control_latents_once_per_unique_image(tmp_path, monkeypatch):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    prompts = [
        {"prompt": "pose one", "control_image_path": str(image_a), "width": 64, "height": 64},
        {"prompt": "pose two", "control_image_path": str(image_a), "width": 64, "height": 64},
        {"prompt": "pose three", "control_image_path": str(image_b), "width": 64, "height": 64},
    ]
    seen_preprocess_paths = []
    image_embed_calls = []
    text_embed_calls = []
    _patch_qwen_prompt_dependencies(monkeypatch, prompts, seen_preprocess_paths, image_embed_calls, text_embed_calls)

    trainer = _recording_trainer()
    sample_parameters = trainer.process_sample_prompts(_qwen_edit_args(), _FakeCPUAccelerator(), str(tmp_path / "prompts.toml"))

    assert len(trainer.loaded_vaes) == 1
    vae, vae_dtype, vae_path = trainer.loaded_vaes[0]
    assert vae_dtype is torch.float32
    assert vae_path == "/tmp/qwen-vae.safetensors"
    assert len(vae.encode_inputs) == 2
    assert torch.equal(sample_parameters[0]["qwen_control_latents"][0], sample_parameters[1]["qwen_control_latents"][0])
    assert not torch.equal(sample_parameters[0]["qwen_control_latents"][0], sample_parameters[2]["qwen_control_latents"][0])
    assert all(param["qwen_control_latents"][0].device.type == "cpu" for param in sample_parameters)
    assert all(param["qwen_control_latents"][0].dtype is torch.bfloat16 for param in sample_parameters)
    assert all("control_image_tensors" not in param for param in sample_parameters)
