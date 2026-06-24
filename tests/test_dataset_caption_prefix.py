import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from musubi_tuner import cache_text_encoder_outputs
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_KREA2,
    AudioDirectoryDatasource,
    AudioJsonlDatasource,
    ImageDirectoryDatasource,
    ImageJsonlDatasource,
    ItemInfo,
    VideoDirectoryDatasource,
    VideoJsonlDatasource,
)


def _touch(path: Path):
    path.write_bytes(b"")


def _generate_dataset_params(user_config: dict):
    args = Namespace(debug_dataset=False)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, args, architecture=ARCHITECTURE_KREA2)
    return blueprint.dataset_group.datasets[0].params


def test_caption_prefix_config_global_fallback_override_and_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dataset = {
            "image_directory": tmpdir,
            "caption_extension": ".txt",
            "cache_directory": str(Path(tmpdir) / "cache"),
        }

        params = _generate_dataset_params({"general": {"caption_prefix": "global "}, "datasets": [base_dataset]})
        assert params.caption_prefix == "global "

        params = _generate_dataset_params(
            {
                "general": {"caption_prefix": "global "},
                "datasets": [{**base_dataset, "caption_prefix": "local "}],
            }
        )
        assert params.caption_prefix == "local "

        params = _generate_dataset_params({"general": {}, "datasets": [base_dataset]})
        assert params.caption_prefix == ""


def test_image_sidecar_caption_prefix_exact_concat_and_missing_fallback():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image = root / "sample.png"
        _touch(image)
        (root / "sample.txt").write_text("a person standing", encoding="utf-8")

        ds = ImageDirectoryDatasource(str(root), ".txt", caption_prefix="middlesplits ")
        assert ds.get_caption(0) == (str(image), "middlesplits a person standing")

        ds = ImageDirectoryDatasource(str(root), ".txt", caption_prefix="middlesplits,")
        assert ds.get_caption(0) == (str(image), "middlesplits,a person standing")

        (root / "sample.txt").write_text("", encoding="utf-8")
        ds = ImageDirectoryDatasource(str(root), ".txt", caption_prefix="middlesplits")
        assert ds.get_caption(0) == (str(image), "middlesplits")

        (root / "sample.txt").unlink()
        ds = ImageDirectoryDatasource(str(root), ".txt", caption_prefix="middlesplits")
        assert len(ds) == 1
        assert ds.get_caption(0) == (str(image), "middlesplits")

        ds = ImageDirectoryDatasource(str(root), ".txt")
        assert len(ds) == 0


def test_video_and_audio_directory_missing_sidecars_use_prefix():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        video = root / "sample.mp4"
        audio = root / "sample.wav"
        _touch(video)
        _touch(audio)

        video_ds = VideoDirectoryDatasource(str(root), ".txt", caption_prefix="scene ")
        assert video_ds.get_caption(0) == (str(video), "scene ")

        audio_ds = AudioDirectoryDatasource(str(root), ".txt", caption_prefix="audio ")
        assert audio_ds.get_caption(0) == (str(audio), "audio ")


def test_jsonl_caption_prefix_for_image_video_and_audio_datasources():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_jsonl = root / "images.jsonl"
        video_jsonl = root / "videos.jsonl"
        audio_jsonl = root / "audio.jsonl"

        image_jsonl.write_text(json.dumps({"image_path": "/tmp/image.png", "caption": "a person"}) + "\n", encoding="utf-8")
        video_jsonl.write_text(json.dumps({"video_path": "/tmp/video.mp4", "caption": "a scene"}) + "\n", encoding="utf-8")
        audio_jsonl.write_text(json.dumps({"audio_path": "/tmp/audio.wav", "caption": ""}) + "\n", encoding="utf-8")

        assert ImageJsonlDatasource(str(image_jsonl), caption_prefix="trigger ").get_caption(0) == (
            "/tmp/image.png",
            "trigger a person",
        )
        assert VideoJsonlDatasource(str(video_jsonl), caption_prefix="trigger,").get_caption(0) == (
            "/tmp/video.mp4",
            "trigger,a scene",
        )
        assert AudioJsonlDatasource(str(audio_jsonl), caption_prefix="audio").get_caption(0) == (
            "/tmp/audio.wav",
            "audio",
        )


def test_text_encoder_cache_caption_match_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        item = ItemInfo("sample", "trigger caption", (0, 0), (0, 0))
        item.text_encoder_output_cache_path = str(root / "sample_te.safetensors")

        save_file({"x": torch.zeros(1)}, item.text_encoder_output_cache_path, metadata={"caption1": "trigger caption"})
        assert cache_text_encoder_outputs.text_encoder_cache_matches_caption(item)

        save_file({"x": torch.zeros(1)}, item.text_encoder_output_cache_path, metadata={"caption1": "old caption"})
        assert not cache_text_encoder_outputs.text_encoder_cache_matches_caption(item)

        save_file({"x": torch.zeros(1)}, item.text_encoder_output_cache_path, metadata={"architecture": "test"})
        assert not cache_text_encoder_outputs.text_encoder_cache_matches_caption(item)
