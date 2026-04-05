from __future__ import annotations

import argparse
import importlib
import os
import time
from pathlib import Path
from typing import Any

import toml
import torch
from accelerate.tracking import GeneralTracker


class TrackioTracker(GeneralTracker):
    name = "trackio"
    requires_logging_directory = False
    main_process_only = False

    def __init__(self, project_name: str, **kwargs) -> None:
        super().__init__()
        self.run_name = project_name
        self._trackio = importlib.import_module("trackio")
        self.run = self._trackio.init(project=project_name, **kwargs)

    @property
    def tracker(self):
        return self.run

    def store_init_configuration(self, values: dict) -> None:
        self._trackio.config.update(values)

    def log(self, values: dict, step: int | None = None, **kwargs) -> None:
        if step is None:
            self._trackio.log(values, **kwargs)
        else:
            self._trackio.log(values, step=step, **kwargs)

    def finish(self) -> None:
        self._trackio.finish()


def prepare_logging(args: argparse.Namespace) -> tuple[str | None, str | None]:
    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    requested_log_with = args.log_with
    if requested_log_with is None:
        accelerator_log_with = "tensorboard" if logging_dir is not None else None
        return logging_dir, accelerator_log_with

    if requested_log_with in ["tensorboard", "all"] and logging_dir is None:
        raise ValueError(
            "logging_dir is required when log_with is tensorboard / Tensorboardを使う場合、logging_dirを指定してください"
        )

    if requested_log_with in ["wandb", "all"]:
        wandb = importlib.import_module("wandb")
        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = logging_dir
        if getattr(args, "wandb_api_key", None) is not None:
            wandb.login(key=args.wandb_api_key)

    if requested_log_with == "trackio":
        importlib.import_module("trackio")
        return logging_dir, None

    return logging_dir, requested_log_with


def build_tracker_init_kwargs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    init_kwargs = toml.load(args.log_tracker_config) if getattr(args, "log_tracker_config", None) is not None else {}

    run_name = getattr(args, "wandb_run_name", None) or getattr(args, "output_name", None)
    if not run_name:
        return init_kwargs

    if getattr(args, "log_with", None) == "trackio":
        init_kwargs.setdefault("trackio", {})
        init_kwargs["trackio"].setdefault("name", run_name)
        init_kwargs["trackio"].setdefault("embed", False)
    else:
        init_kwargs.setdefault("wandb", {})
        init_kwargs["wandb"].setdefault("name", run_name)

    return init_kwargs


def resolve_tracker_name(args: argparse.Namespace, default_tracker_name: str) -> str:
    explicit_tracker_name = getattr(args, "log_tracker_name", None)
    if explicit_tracker_name:
        return explicit_tracker_name

    network_module = getattr(args, "network_module", None)
    if network_module:
        return network_module

    return default_tracker_name


def init_experiment_trackers(
    accelerator,
    args: argparse.Namespace,
    default_tracker_name: str,
    config: dict | None = None,
) -> None:
    if not accelerator.is_main_process:
        return

    tracker_name = resolve_tracker_name(args, default_tracker_name)
    init_kwargs = build_tracker_init_kwargs(args)

    if getattr(args, "log_with", None) == "trackio":
        trackio_kwargs = dict(init_kwargs.get("trackio", {}))
        trackio_kwargs.setdefault("embed", False)
        tracker = TrackioTracker(tracker_name, **trackio_kwargs)
        accelerator.trackers.append(tracker)
        if config is not None:
            tracker.store_init_configuration(config)
        return

    accelerator.init_trackers(tracker_name, config=config, init_kwargs=init_kwargs)


def _find_tracker(accelerator, tracker_name: str):
    for tracker in getattr(accelerator, "trackers", []):
        if tracker.name == tracker_name:
            return tracker
    return None


def log_named_artifact(accelerator, name: str, path: str | Path, artifact_type: str, step: int | None = None) -> None:
    path = Path(path)

    def _artifact_factory(module):
        if artifact_type == "image":
            return module.Image
        if artifact_type == "video":
            return module.Video
        if artifact_type == "audio":
            return module.Audio
        raise ValueError(f"Unsupported artifact type: {artifact_type}")

    wandb_tracker = _find_tracker(accelerator, "wandb")
    if wandb_tracker is not None:
        wandb = importlib.import_module("wandb")
        factory = _artifact_factory(wandb)
        wandb_tracker.log({name: factory(path)}, step=step)

    trackio_tracker = _find_tracker(accelerator, "trackio")
    if trackio_tracker is not None:
        trackio = importlib.import_module("trackio")
        factory = _artifact_factory(trackio)
        trackio_tracker.log({name: factory(path)}, step=step)


def log_histogram(accelerator, name: str, values: torch.Tensor, step: int | None = None) -> None:
    values_cpu = values.detach().cpu()
    values_numpy = values_cpu.numpy()

    for tracker in getattr(accelerator, "trackers", []):
        if tracker.name == "tensorboard":
            tracker.writer.add_histogram(name, values_cpu, step)
        elif tracker.name == "wandb":
            wandb = importlib.import_module("wandb")
            tracker.log({name: wandb.Histogram(values_numpy)}, step=step)
        elif tracker.name == "trackio":
            trackio = importlib.import_module("trackio")
            tracker.log({name: trackio.Histogram(values_numpy)}, step=step)
