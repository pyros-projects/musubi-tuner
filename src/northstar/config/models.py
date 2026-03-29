"""Typed config models for Northstar."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfigSection:
    name: str
    architecture: str
    seed: int | None = None


@dataclass(frozen=True)
class ModelConfigSection:
    version: str
    dit: str
    vae: str | None = None
    text_encoder: str | None = None
    sdpa: bool = False
    fp8_base: bool = False
    fp8_scaled: bool = False


@dataclass(frozen=True)
class AccelerateConfigSection:
    num_cpu_threads_per_process: int | None = None
    mixed_precision: str | None = None


@dataclass(frozen=True)
class TrainingConfigSection:
    timestep_sampling: str | None = None
    weighting_scheme: str | None = None
    gradient_checkpointing: bool = False
    learning_rate: float | None = None
    max_grad_norm: float | None = None
    max_train_epochs: int | None = None
    save_every_n_epochs: int | None = None
    max_data_loader_n_workers: int | None = None
    persistent_data_loader_workers: bool = False


@dataclass(frozen=True)
class OptimizerConfigSection:
    type: str | None = None
    lr_scheduler: str | None = None
    args: dict[str, bool | int | float | str] | None = None


@dataclass(frozen=True)
class NetworkConfigSection:
    module: str | None = None
    dim: int | None = None
    alpha: float | None = None


@dataclass(frozen=True)
class PerformanceConfigSection:
    compile_dynamic: bool = False
    compile_prewarm: bool = False
    cuda_allow_tf32: bool = False


@dataclass(frozen=True)
class OutputConfigSection:
    dir: str | None = None
    name: str | None = None
    logging_dir: str | None = None


@dataclass(frozen=True)
class SamplingLoraConfig:
    path: str
    multiplier: float


@dataclass(frozen=True)
class SamplingPromptConfig:
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    guidance: float
    negative_prompt: str


@dataclass(frozen=True)
class SamplingConfigSection:
    enabled: bool = False
    sample_at_first: bool = False
    sample_every_n_steps: int | None = None
    loras: list[SamplingLoraConfig] | None = None
    prompts: list[SamplingPromptConfig] | None = None


@dataclass(frozen=True)
class DatasetSourceConfig:
    image_directory: str
    num_repeats: int


@dataclass(frozen=True)
class DatasetGroupConfig:
    resolution: tuple[int, int]
    batch_size: int
    enable_bucket: bool
    bucket_no_upscale: bool
    cache_directory: str
    caption_extension: str
    sources: list[DatasetSourceConfig]


@dataclass(frozen=True)
class DatasetConfigSection:
    groups: list[DatasetGroupConfig] | None = None


@dataclass(frozen=True)
class RunConfig:
    run: RunConfigSection
    model: ModelConfigSection
    accelerate: AccelerateConfigSection | None = None
    training: TrainingConfigSection | None = None
    optimizer: OptimizerConfigSection | None = None
    network: NetworkConfigSection | None = None
    performance: PerformanceConfigSection | None = None
    output: OutputConfigSection | None = None
    sampling: SamplingConfigSection | None = None
    dataset: DatasetConfigSection | None = None
