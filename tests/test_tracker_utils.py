from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from musubi_tuner.utils import tracker_utils


class _FakeTrackioConfig:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update(self, values: dict) -> None:
        self.updates.append(values)


class _FakeTrackioImage:
    def __init__(self, value, caption=None) -> None:
        self.value = value
        self.caption = caption


class _FakeTrackioVideo:
    def __init__(self, value, caption=None, fps=None, format=None) -> None:
        self.value = value
        self.caption = caption
        self.fps = fps
        self.format = format


class _FakeTrackioHistogram:
    def __init__(self, sequence=None, np_histogram=None, num_bins=64) -> None:
        self.sequence = sequence
        self.np_histogram = np_histogram
        self.num_bins = num_bins


class _FakeTrackioModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("trackio")
        self.Image = _FakeTrackioImage
        self.Video = _FakeTrackioVideo
        self.Histogram = _FakeTrackioHistogram
        self.config = _FakeTrackioConfig()
        self.init_calls: list[dict] = []
        self.logged: list[tuple[dict, int | None, dict]] = []
        self.finish_calls = 0

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return types.SimpleNamespace(run_id="trackio-run")

    def log(self, values: dict, step: int | None = None, **kwargs) -> None:
        self.logged.append((values, step, kwargs))

    def finish(self) -> None:
        self.finish_calls += 1


class _FakeTrackerAccelerator:
    def __init__(self) -> None:
        self.trackers = []
        self.init_trackers_calls = []
        self.is_main_process = True

    def init_trackers(self, project_name: str, config: dict | None = None, init_kwargs: dict | None = None) -> None:
        self.init_trackers_calls.append((project_name, config, init_kwargs))


class TrackerUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_trackio = _FakeTrackioModule()
        self.trackio_modules = {"trackio": self.fake_trackio}

    def test_prepare_logging_uses_custom_trackio_path(self) -> None:
        args = types.SimpleNamespace(
            logging_dir=tempfile.mkdtemp(),
            log_prefix="logs-",
            log_with="trackio",
            wandb_api_key=None,
        )

        with mock.patch.dict(sys.modules, self.trackio_modules, clear=False):
            logging_dir, accelerator_log_with = tracker_utils.prepare_logging(args)

        self.assertIsNone(accelerator_log_with)
        self.assertTrue(logging_dir.startswith(args.logging_dir))

    def test_init_experiment_trackers_registers_trackio_tracker(self) -> None:
        accelerator = _FakeTrackerAccelerator()
        args = types.SimpleNamespace(
            log_with="trackio",
            log_tracker_name="musubi-trackio",
            wandb_run_name="run-123",
            log_tracker_config=None,
        )

        with mock.patch.dict(sys.modules, self.trackio_modules, clear=False):
            tracker_utils.init_experiment_trackers(
                accelerator,
                args,
                default_tracker_name="network_train",
                config={"lr": 1e-4},
            )

        self.assertEqual(accelerator.init_trackers_calls, [])
        self.assertEqual(len(accelerator.trackers), 1)
        self.assertEqual(accelerator.trackers[0].name, "trackio")
        self.assertEqual(
            self.fake_trackio.init_calls,
            [{"project": "musubi-trackio", "name": "run-123", "embed": False}],
        )
        self.assertEqual(self.fake_trackio.config.updates, [{"lr": 1e-4}])

    def test_init_experiment_trackers_uses_output_name_as_default_run_name_for_trackio(self) -> None:
        accelerator = _FakeTrackerAccelerator()
        args = types.SimpleNamespace(
            log_with="trackio",
            log_tracker_name="musubi-trackio",
            wandb_run_name=None,
            output_name="sunset-lora",
            log_tracker_config=None,
        )

        with mock.patch.dict(sys.modules, self.trackio_modules, clear=False):
            tracker_utils.init_experiment_trackers(
                accelerator,
                args,
                default_tracker_name="network_train",
                config=None,
            )

        self.assertEqual(
            self.fake_trackio.init_calls,
            [{"project": "musubi-trackio", "name": "sunset-lora", "embed": False}],
        )

    def test_init_experiment_trackers_uses_network_module_as_default_project_name(self) -> None:
        accelerator = _FakeTrackerAccelerator()
        args = types.SimpleNamespace(
            log_with="trackio",
            log_tracker_name=None,
            network_module="networks.lora_zimage",
            wandb_run_name=None,
            output_name="sunset-lora",
            log_tracker_config=None,
        )

        with mock.patch.dict(sys.modules, self.trackio_modules, clear=False):
            tracker_utils.init_experiment_trackers(
                accelerator,
                args,
                default_tracker_name="network_train",
                config=None,
            )

        self.assertEqual(
            self.fake_trackio.init_calls,
            [{"project": "networks.lora_zimage", "name": "sunset-lora", "embed": False}],
        )

    def test_log_named_artifact_logs_trackio_images_and_videos(self) -> None:
        accelerator = _FakeTrackerAccelerator()
        args = types.SimpleNamespace(
            log_with="trackio",
            log_tracker_name=None,
            wandb_run_name=None,
            log_tracker_config=None,
        )

        with mock.patch.dict(sys.modules, self.trackio_modules, clear=False):
            tracker_utils.init_experiment_trackers(accelerator, args, default_tracker_name="network_train")
            tracker_utils.log_named_artifact(accelerator, "sample_0", Path("/tmp/sample.png"), "image", step=3)
            tracker_utils.log_named_artifact(accelerator, "sample_1", Path("/tmp/sample.mp4"), "video", step=4)

        self.assertEqual(len(self.fake_trackio.logged), 2)
        image_log, video_log = self.fake_trackio.logged
        self.assertIsInstance(image_log[0]["sample_0"], _FakeTrackioImage)
        self.assertEqual(image_log[0]["sample_0"].value, Path("/tmp/sample.png"))
        self.assertEqual(image_log[1], 3)
        self.assertIsInstance(video_log[0]["sample_1"], _FakeTrackioVideo)
        self.assertEqual(video_log[0]["sample_1"].value, Path("/tmp/sample.mp4"))
        self.assertEqual(video_log[1], 4)

    def test_log_histogram_logs_trackio_histogram(self) -> None:
        accelerator = _FakeTrackerAccelerator()
        args = types.SimpleNamespace(
            log_with="trackio",
            log_tracker_name=None,
            wandb_run_name=None,
            log_tracker_config=None,
        )

        with mock.patch.dict(sys.modules, self.trackio_modules, clear=False):
            tracker_utils.init_experiment_trackers(accelerator, args, default_tracker_name="network_train")
            tracker_utils.log_histogram(accelerator, "lr/automagic_lrs", torch.tensor([1.0, 2.0, 3.0]), step=9)

        self.assertEqual(len(self.fake_trackio.logged), 1)
        histogram_log = self.fake_trackio.logged[0]
        self.assertIsInstance(histogram_log[0]["lr/automagic_lrs"], _FakeTrackioHistogram)
        self.assertEqual(histogram_log[1], 9)


if __name__ == "__main__":
    unittest.main()
