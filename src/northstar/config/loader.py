"""Config loading for Northstar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from northstar.config.models import AccelerateConfigSection
from northstar.config.models import DatasetConfigSection
from northstar.config.models import DatasetGroupConfig
from northstar.config.models import DatasetSourceConfig
from northstar.config.models import ModelConfigSection
from northstar.config.models import NetworkConfigSection
from northstar.config.models import OptimizerConfigSection
from northstar.config.models import OutputConfigSection
from northstar.config.models import PerformanceConfigSection
from northstar.config.models import RunConfig
from northstar.config.models import RunConfigSection
from northstar.config.models import SamplingConfigSection
from northstar.config.models import SamplingLoraConfig
from northstar.config.models import SamplingPromptConfig
from northstar.config.models import TrainingConfigSection

ALLOWED_ARCHITECTURES = {"flux2", "zimage"}
ALLOWED_MIXED_PRECISION = {"no", "fp16", "bf16"}


def _join_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _ensure_allowed_keys(data: dict[str, Any], allowed_keys: set[str], path: str) -> None:
    for key in data:
        if key not in allowed_keys:
            raise ValueError(f"Unknown config key: {_join_path(path, key)}")


def _require_table(data: dict[str, Any], key: str, path: str = "") -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required table: {_join_path(path, key)}")
    return value


def _require_string(data: dict[str, Any], key: str, path: str = "") -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing required string field: {_join_path(path, key)}")
    return value


def _optional_table(data: dict[str, Any], key: str, path: str = "") -> dict[str, Any] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Expected table for key: {_join_path(path, key)}")
    return value


def _optional_bool(data: dict[str, Any], key: str, path: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean field: {_join_path(path, key)}")
    return value


def _optional_int(data: dict[str, Any], key: str, path: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Expected integer field: {_join_path(path, key)}")
    return value


def _require_int(data: dict[str, Any], key: str, path: str) -> int:
    value = _optional_int(data, key, path)
    if value is None:
        raise ValueError(f"Missing required integer field: {_join_path(path, key)}")
    return value


def _optional_float(data: dict[str, Any], key: str, path: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Expected numeric field: {_join_path(path, key)}")
    return float(value)


def _require_float(data: dict[str, Any], key: str, path: str) -> float:
    value = _optional_float(data, key, path)
    if value is None:
        raise ValueError(f"Missing required numeric field: {_join_path(path, key)}")
    return value


def _optional_string(data: dict[str, Any], key: str, path: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected string field: {_join_path(path, key)}")
    return value


def _require_one_of(value: str, allowed_values: set[str], path: str) -> str:
    if value not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(f"Invalid value for {path}: {value!r}. Expected one of: {allowed}")
    return value


def _validate_primitive_map(data: dict[str, Any], path: str) -> dict[str, bool | int | float | str]:
    validated: dict[str, bool | int | float | str] = {}
    for key, value in data.items():
        current_path = _join_path(path, key)
        if isinstance(value, bool):
            validated[key] = value
        elif isinstance(value, int):
            validated[key] = value
        elif isinstance(value, float):
            validated[key] = value
        elif isinstance(value, str):
            validated[key] = value
        else:
            raise ValueError(f"Expected primitive value for {current_path}")
    return validated


def _parse_accelerate(table: dict[str, Any] | None) -> AccelerateConfigSection | None:
    if table is None:
        return None
    _ensure_allowed_keys(table, {"num_cpu_threads_per_process", "mixed_precision"}, "accelerate")
    return AccelerateConfigSection(
        num_cpu_threads_per_process=_optional_int(table, "num_cpu_threads_per_process", "accelerate"),
        mixed_precision=(
            _require_one_of(mixed_precision, ALLOWED_MIXED_PRECISION, "accelerate.mixed_precision")
            if (mixed_precision := _optional_string(table, "mixed_precision", "accelerate")) is not None
            else None
        ),
    )


def _parse_training(table: dict[str, Any] | None) -> TrainingConfigSection | None:
    if table is None:
        return None
    _ensure_allowed_keys(
        table,
        {
            "timestep_sampling",
            "weighting_scheme",
            "gradient_checkpointing",
            "learning_rate",
            "max_grad_norm",
            "max_train_epochs",
            "save_every_n_epochs",
            "max_data_loader_n_workers",
            "persistent_data_loader_workers",
        },
        "training",
    )
    return TrainingConfigSection(
        timestep_sampling=_optional_string(table, "timestep_sampling", "training"),
        weighting_scheme=_optional_string(table, "weighting_scheme", "training"),
        gradient_checkpointing=_optional_bool(table, "gradient_checkpointing", "training"),
        learning_rate=_optional_float(table, "learning_rate", "training"),
        max_grad_norm=_optional_float(table, "max_grad_norm", "training"),
        max_train_epochs=_optional_int(table, "max_train_epochs", "training"),
        save_every_n_epochs=_optional_int(table, "save_every_n_epochs", "training"),
        max_data_loader_n_workers=_optional_int(table, "max_data_loader_n_workers", "training"),
        persistent_data_loader_workers=_optional_bool(table, "persistent_data_loader_workers", "training"),
    )


def _parse_optimizer(table: dict[str, Any] | None) -> OptimizerConfigSection | None:
    if table is None:
        return None
    _ensure_allowed_keys(table, {"type", "lr_scheduler", "args"}, "optimizer")
    raw_args = table.get("args")
    if raw_args is not None and not isinstance(raw_args, dict):
        raise ValueError("Expected table for key: optimizer.args")
    return OptimizerConfigSection(
        type=_optional_string(table, "type", "optimizer"),
        lr_scheduler=_optional_string(table, "lr_scheduler", "optimizer"),
        args=_validate_primitive_map(raw_args, "optimizer.args") if raw_args is not None else None,
    )


def _parse_network(table: dict[str, Any] | None) -> NetworkConfigSection | None:
    if table is None:
        return None
    _ensure_allowed_keys(table, {"module", "dim", "alpha"}, "network")
    return NetworkConfigSection(
        module=_optional_string(table, "module", "network"),
        dim=_optional_int(table, "dim", "network"),
        alpha=_optional_float(table, "alpha", "network"),
    )


def _parse_performance(table: dict[str, Any] | None) -> PerformanceConfigSection | None:
    if table is None:
        return None
    _ensure_allowed_keys(table, {"compile_dynamic", "compile_prewarm", "cuda_allow_tf32"}, "performance")
    return PerformanceConfigSection(
        compile_dynamic=_optional_bool(table, "compile_dynamic", "performance"),
        compile_prewarm=_optional_bool(table, "compile_prewarm", "performance"),
        cuda_allow_tf32=_optional_bool(table, "cuda_allow_tf32", "performance"),
    )


def _parse_output(table: dict[str, Any] | None) -> OutputConfigSection | None:
    if table is None:
        return None
    _ensure_allowed_keys(table, {"dir", "name", "logging_dir"}, "output")
    return OutputConfigSection(
        dir=_optional_string(table, "dir", "output"),
        name=_optional_string(table, "name", "output"),
        logging_dir=_optional_string(table, "logging_dir", "output"),
    )


def _parse_sampling(table: dict[str, Any] | None) -> SamplingConfigSection | None:
    if table is None:
        return None
    _ensure_allowed_keys(table, {"enabled", "sample_at_first", "sample_every_n_steps", "loras", "prompts"}, "sampling")

    raw_loras = table.get("loras", [])
    if not isinstance(raw_loras, list):
        raise ValueError("Expected array for key: sampling.loras")
    loras = []
    for index, raw_lora in enumerate(raw_loras):
        _ensure_allowed_keys(raw_lora, {"path", "multiplier"}, f"sampling.loras[{index}]")
        multiplier = raw_lora.get("multiplier")
        if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool):
            raise ValueError(f"Expected numeric field: sampling.loras[{index}].multiplier")
        loras.append(
            SamplingLoraConfig(
                path=_require_string(raw_lora, "path", f"sampling.loras[{index}]"),
                multiplier=float(multiplier),
            )
        )

    raw_prompts = table.get("prompts", [])
    if not isinstance(raw_prompts, list):
        raise ValueError("Expected array for key: sampling.prompts")
    prompts = []
    for index, raw_prompt in enumerate(raw_prompts):
        prompt_path = f"sampling.prompts[{index}]"
        _ensure_allowed_keys(raw_prompt, {"prompt", "width", "height", "steps", "seed", "guidance", "negative_prompt"}, prompt_path)
        prompts.append(
            SamplingPromptConfig(
                prompt=_require_string(raw_prompt, "prompt", prompt_path),
                width=_require_int(raw_prompt, "width", prompt_path),
                height=_require_int(raw_prompt, "height", prompt_path),
                steps=_require_int(raw_prompt, "steps", prompt_path),
                seed=_require_int(raw_prompt, "seed", prompt_path),
                guidance=_require_float(raw_prompt, "guidance", prompt_path),
                negative_prompt=_require_string(raw_prompt, "negative_prompt", prompt_path),
            )
        )

    return SamplingConfigSection(
        enabled=_optional_bool(table, "enabled", "sampling"),
        sample_at_first=_optional_bool(table, "sample_at_first", "sampling"),
        sample_every_n_steps=_optional_int(table, "sample_every_n_steps", "sampling"),
        loras=loras,
        prompts=prompts,
    )


def _parse_dataset(table: dict[str, Any] | None) -> DatasetConfigSection | None:
    if table is None:
        return None
    _ensure_allowed_keys(table, {"groups"}, "dataset")

    raw_groups = table.get("groups", [])
    if not isinstance(raw_groups, list):
        raise ValueError("Expected array for key: dataset.groups")

    groups: list[DatasetGroupConfig] = []
    for index, raw_group in enumerate(raw_groups):
        group_path = f"dataset.groups[{index}]"
        _ensure_allowed_keys(
            raw_group,
            {"resolution", "batch_size", "enable_bucket", "bucket_no_upscale", "cache_directory", "caption_extension", "sources"},
            group_path,
        )
        resolution = raw_group.get("resolution")
        if not isinstance(resolution, list) or len(resolution) != 2 or not all(isinstance(value, int) for value in resolution):
            raise ValueError("Expected two-item integer array for key: dataset.groups[].resolution")
        raw_sources = raw_group.get("sources", [])
        if not isinstance(raw_sources, list):
            raise ValueError("Expected array for key: dataset.groups[].sources")
        sources = []
        for source_index, raw_source in enumerate(raw_sources):
            source_path = f"{group_path}.sources[{source_index}]"
            _ensure_allowed_keys(raw_source, {"image_directory", "num_repeats"}, source_path)
            sources.append(
                DatasetSourceConfig(
                    image_directory=_require_string(raw_source, "image_directory", source_path),
                    num_repeats=_optional_int(raw_source, "num_repeats", source_path) or 1,
                )
            )
        groups.append(
            DatasetGroupConfig(
                resolution=(resolution[0], resolution[1]),
                batch_size=_require_int(raw_group, "batch_size", group_path),
                enable_bucket=_optional_bool(raw_group, "enable_bucket", group_path),
                bucket_no_upscale=_optional_bool(raw_group, "bucket_no_upscale", group_path),
                cache_directory=_require_string(raw_group, "cache_directory", group_path),
                caption_extension=_require_string(raw_group, "caption_extension", group_path),
                sources=sources,
            )
        )

    return DatasetConfigSection(groups=groups)


def load_run_config(path: str | Path) -> RunConfig:
    raw_config = toml.load(Path(path))
    _ensure_allowed_keys(raw_config, {"run", "accelerate", "model", "training", "optimizer", "network", "performance", "output", "sampling", "dataset"}, "")

    run_table = _require_table(raw_config, "run")
    model_table = _require_table(raw_config, "model")
    _ensure_allowed_keys(run_table, {"name", "architecture", "seed"}, "run")
    _ensure_allowed_keys(model_table, {"version", "dit", "vae", "text_encoder", "sdpa", "fp8_base", "fp8_scaled"}, "model")

    return RunConfig(
        run=RunConfigSection(
            name=_require_string(run_table, "name", "run"),
            architecture=_require_one_of(_require_string(run_table, "architecture", "run"), ALLOWED_ARCHITECTURES, "run.architecture"),
            seed=_optional_int(run_table, "seed", "run"),
        ),
        model=ModelConfigSection(
            version=_require_string(model_table, "version", "model"),
            dit=_require_string(model_table, "dit", "model"),
            vae=_optional_string(model_table, "vae", "model"),
            text_encoder=_optional_string(model_table, "text_encoder", "model"),
            sdpa=_optional_bool(model_table, "sdpa", "model"),
            fp8_base=_optional_bool(model_table, "fp8_base", "model"),
            fp8_scaled=_optional_bool(model_table, "fp8_scaled", "model"),
        ),
        accelerate=_parse_accelerate(_optional_table(raw_config, "accelerate")),
        training=_parse_training(_optional_table(raw_config, "training")),
        optimizer=_parse_optimizer(_optional_table(raw_config, "optimizer")),
        network=_parse_network(_optional_table(raw_config, "network")),
        performance=_parse_performance(_optional_table(raw_config, "performance")),
        output=_parse_output(_optional_table(raw_config, "output")),
        sampling=_parse_sampling(_optional_table(raw_config, "sampling")),
        dataset=_parse_dataset(_optional_table(raw_config, "dataset")),
    )
