import json
import os
from argparse import Namespace

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner import cache_text_encoder_outputs
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer, generate_dataset_group_by_blueprint
from musubi_tuner.dataset.datasources import (
    ImageDirectoryDatasource,
    ImageJsonlDatasource,
    VideoDirectoryDatasource,
    VideoJsonlDatasource,
)
from musubi_tuner.dataset.image_video_dataset import ItemInfo


def _blueprint(config):
    return BlueprintGenerator(ConfigSanitizer()).generate(
        config,
        Namespace(debug_dataset=False),
        architecture=ARCHITECTURE_MINIMAX_H3,
    )


def _dataset_params(config):
    return _blueprint(config).dataset_group.datasets[0].params


def test_caption_prefix_config_fallback_override_and_default(tmp_path):
    dataset = {
        "image_directory": str(tmp_path),
        "caption_extension": ".txt",
        "cache_directory": str(tmp_path / "cache"),
    }

    config = {"general": {"caption_prefix": "global "}, "datasets": [dataset]}
    params = _dataset_params(config)
    assert params.caption_prefix == "global "
    runtime_dataset = generate_dataset_group_by_blueprint(_blueprint(config).dataset_group).datasets[0]
    assert runtime_dataset.caption_prefix == "global "
    assert runtime_dataset.datasource.caption_prefix == "global "

    params = _dataset_params(
        {
            "general": {"caption_prefix": "global "},
            "datasets": [{**dataset, "caption_prefix": "local "}],
        }
    )
    assert params.caption_prefix == "local "

    params = _dataset_params({"general": {}, "datasets": [dataset]})
    assert params.caption_prefix == ""


def test_directory_caption_prefix_and_missing_sidecar_fallback(tmp_path):
    image = tmp_path / "sample.png"
    video = tmp_path / "sample.mp4"
    image.touch()
    video.touch()
    image.with_suffix(".txt").write_text("a person standing", encoding="utf-8")

    image_source = ImageDirectoryDatasource(str(tmp_path), ".txt", caption_prefix="trigger, ")
    assert image_source.get_caption(0) == (str(image), "trigger, a person standing")

    image.with_suffix(".txt").unlink()
    image_source = ImageDirectoryDatasource(str(tmp_path), ".txt", caption_prefix="trigger")
    assert image_source.get_caption(0) == (str(image), "trigger")
    assert len(ImageDirectoryDatasource(str(tmp_path), ".txt")) == 0

    video_source = VideoDirectoryDatasource(str(tmp_path), ".txt", caption_prefix="scene ")
    assert video_source.get_caption(0) == (str(video), "scene ")

    with pytest.raises(ValueError, match="caption_extension is required"):
        ImageDirectoryDatasource(str(tmp_path), multiple_target=True, caption_prefix="trigger")


def test_jsonl_caption_prefix_for_image_and_video(tmp_path):
    image_jsonl = tmp_path / "images.jsonl"
    video_jsonl = tmp_path / "videos.jsonl"
    image_jsonl.write_text(json.dumps({"image_path": "/tmp/image.png", "caption": "a person"}) + "\n", encoding="utf-8")
    video_jsonl.write_text(json.dumps({"video_path": "/tmp/video.mp4", "caption": "a scene"}) + "\n", encoding="utf-8")

    assert ImageJsonlDatasource(str(image_jsonl), caption_prefix="trigger ").get_caption(0) == (
        "/tmp/image.png",
        "trigger a person",
    )
    assert VideoJsonlDatasource(str(video_jsonl), caption_prefix="trigger,").get_caption(0) == (
        "/tmp/video.mp4",
        "trigger,a scene",
    )


def test_text_encoder_cache_caption_match_detection(tmp_path):
    item = ItemInfo("sample", "trigger caption", (0, 0), (0, 0))
    item.text_encoder_output_cache_path = str(tmp_path / "sample_te.safetensors")

    save_file({"x": torch.zeros(1)}, item.text_encoder_output_cache_path, metadata={"caption1": item.caption})
    assert cache_text_encoder_outputs.text_encoder_cache_matches_caption(item)

    save_file({"x": torch.zeros(1)}, item.text_encoder_output_cache_path, metadata={"caption1": "old caption"})
    assert not cache_text_encoder_outputs.text_encoder_cache_matches_caption(item)

    save_file({"x": torch.zeros(1)}, item.text_encoder_output_cache_path)
    assert not cache_text_encoder_outputs.text_encoder_cache_matches_caption(item)


def test_stale_text_encoder_cache_is_rebuilt_for_every_encoder_pass(tmp_path):
    item = ItemInfo("sample", "new caption", (0, 0), (0, 0))
    item.text_encoder_output_cache_path = str(tmp_path / "sample_te.safetensors")
    save_file({"x": torch.zeros(1)}, item.text_encoder_output_cache_path, metadata={"caption1": "old caption"})

    class Dataset:
        def retrieve_text_encoder_output_cache_batches(self, _num_workers):
            yield [item]

    cache_files = [{os.path.normpath(item.text_encoder_output_cache_path)}]
    cache_paths = [set()]
    existed_before_encode = []

    def encode(_batch):
        existed_before_encode.append(os.path.exists(item.text_encoder_output_cache_path))
        save_file({"x": torch.zeros(1)}, item.text_encoder_output_cache_path, metadata={"caption1": item.caption})

    for _ in range(2):
        cache_text_encoder_outputs.process_text_encoder_batches(
            1,
            True,
            1,
            [Dataset()],
            cache_files,
            cache_paths,
            encode,
        )

    assert existed_before_encode == [False, True]
