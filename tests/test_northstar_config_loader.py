from __future__ import annotations

import tempfile
import textwrap
import typing
import unittest
from pathlib import Path


class NorthstarConfigLoaderTest(unittest.TestCase):
    def test_loads_minimal_run_and_model_config(self) -> None:
        from northstar.config.loader import load_run_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "minimal.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    name = "flux2-smoke"
                    architecture = "flux2"

                    [model]
                    version = "klein-base-9b"
                    dit = "/models/flux2.safetensors"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_run_config(config_path)

        self.assertEqual(config.run.name, "flux2-smoke")
        self.assertEqual(config.run.architecture, "flux2")
        self.assertEqual(config.model.version, "klein-base-9b")
        self.assertEqual(config.model.dit, "/models/flux2.safetensors")

    def test_loads_full_smoke_config_shape(self) -> None:
        from northstar.config.loader import load_run_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "full.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    name = "bb_flux"
                    architecture = "flux2"
                    seed = 42

                    [accelerate]
                    num_cpu_threads_per_process = 1
                    mixed_precision = "bf16"

                    [model]
                    version = "klein-base-9b"
                    dit = "/models/flux2.safetensors"
                    vae = "/models/ae.safetensors"
                    text_encoder = "/models/text-encoder.safetensors"
                    sdpa = true
                    fp8_base = true
                    fp8_scaled = true

                    [training]
                    timestep_sampling = "flux2_shift"
                    weighting_scheme = "none"
                    gradient_checkpointing = true
                    learning_rate = 0.0002
                    max_grad_norm = 0
                    max_train_epochs = 60
                    save_every_n_epochs = 1
                    max_data_loader_n_workers = 2
                    persistent_data_loader_workers = true

                    [optimizer]
                    type = "adafactor"
                    lr_scheduler = "constant"

                    [optimizer.args]
                    scale_parameter = false
                    relative_step = false
                    warmup_init = false

                    [network]
                    module = "networks.lora_flux_2"
                    dim = 16
                    alpha = 16

                    [performance]
                    compile_dynamic = true
                    compile_prewarm = true
                    cuda_allow_tf32 = true

                    [output]
                    dir = "/tmp/out"
                    name = "bb_flux"
                    logging_dir = "/tmp/out"

                    [sampling]
                    enabled = true
                    sample_at_first = true
                    sample_every_n_steps = 50

                    [[sampling.loras]]
                    path = "/models/sample-lora.safetensors"
                    multiplier = 0.6

                    [[sampling.prompts]]
                    prompt = "a portrait"
                    width = 832
                    height = 1216
                    steps = 6
                    seed = 6666
                    guidance = 4
                    negative_prompt = " "

                    [[dataset.groups]]
                    resolution = [320, 320]
                    batch_size = 4
                    enable_bucket = true
                    bucket_no_upscale = false
                    cache_directory = "cache/group1"
                    caption_extension = ".txt"

                    [[dataset.groups.sources]]
                    image_directory = "/datasets/group1"
                    num_repeats = 1
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_run_config(config_path)

        self.assertEqual(config.run.seed, 42)
        self.assertEqual(config.accelerate.num_cpu_threads_per_process, 1)
        self.assertEqual(config.model.vae, "/models/ae.safetensors")
        self.assertEqual(config.training.max_train_epochs, 60)
        self.assertEqual(config.optimizer.args["scale_parameter"], False)
        self.assertEqual(config.network.dim, 16)
        self.assertTrue(config.performance.compile_prewarm)
        self.assertEqual(config.output.name, "bb_flux")
        self.assertEqual(config.sampling.prompts[0].width, 832)
        self.assertEqual(config.dataset.groups[0].sources[0].image_directory, "/datasets/group1")

    def test_rejects_unknown_config_key(self) -> None:
        from northstar.config.loader import load_run_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    name = "flux2-smoke"
                    architecture = "flux2"
                    surprise = "nope"

                    [model]
                    version = "klein-base-9b"
                    dit = "/models/flux2.safetensors"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "run.surprise"):
                load_run_config(config_path)

    def test_rejects_invalid_type_with_field_name(self) -> None:
        from northstar.config.loader import load_run_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid-type.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    name = "flux2-smoke"
                    architecture = "flux2"

                    [model]
                    version = "klein-base-9b"
                    dit = "/models/flux2.safetensors"

                    [performance]
                    compile_prewarm = "yes"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "performance.compile_prewarm"):
                load_run_config(config_path)

    def test_rejects_prompt_missing_required_numeric_fields(self) -> None:
        from northstar.config.loader import load_run_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "missing-prompt-width.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    name = "flux2-smoke"
                    architecture = "flux2"

                    [model]
                    version = "klein-base-9b"
                    dit = "/models/flux2.safetensors"

                    [sampling]
                    enabled = true
                    sample_at_first = true

                    [[sampling.prompts]]
                    prompt = "a portrait"
                    height = 1216
                    steps = 6
                    seed = 6666
                    guidance = 4
                    negative_prompt = " "
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "sampling.prompts\\[0\\]\\.width"):
                load_run_config(config_path)

    def test_rejects_invalid_architecture_value(self) -> None:
        from northstar.config.loader import load_run_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid-architecture.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    name = "flux2-smoke"
                    architecture = "flux"

                    [model]
                    version = "klein-base-9b"
                    dit = "/models/flux2.safetensors"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "run.architecture"):
                load_run_config(config_path)

    def test_rejects_invalid_mixed_precision_value(self) -> None:
        from northstar.config.loader import load_run_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid-mixed-precision.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    name = "flux2-smoke"
                    architecture = "flux2"

                    [accelerate]
                    mixed_precision = "weird"

                    [model]
                    version = "klein-base-9b"
                    dit = "/models/flux2.safetensors"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "accelerate.mixed_precision"):
                load_run_config(config_path)

    def test_dataset_num_repeats_defaults_to_one(self) -> None:
        from northstar.config.loader import load_run_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "dataset-default-repeats.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    name = "flux2-smoke"
                    architecture = "flux2"

                    [model]
                    version = "klein-base-9b"
                    dit = "/models/flux2.safetensors"

                    [[dataset.groups]]
                    resolution = [320, 320]
                    batch_size = 4
                    enable_bucket = true
                    bucket_no_upscale = false
                    cache_directory = "cache/group1"
                    caption_extension = ".txt"

                    [[dataset.groups.sources]]
                    image_directory = "/datasets/group1"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_run_config(config_path)

        self.assertEqual(config.dataset.groups[0].sources[0].num_repeats, 1)

    def test_rejects_dataset_group_missing_batch_size(self) -> None:
        from northstar.config.loader import load_run_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "missing-batch-size.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    name = "flux2-smoke"
                    architecture = "flux2"

                    [model]
                    version = "klein-base-9b"
                    dit = "/models/flux2.safetensors"

                    [[dataset.groups]]
                    resolution = [320, 320]
                    enable_bucket = true
                    bucket_no_upscale = false
                    cache_directory = "cache/group1"
                    caption_extension = ".txt"

                    [[dataset.groups.sources]]
                    image_directory = "/datasets/group1"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "dataset.groups\\[0\\]\\.batch_size"):
                load_run_config(config_path)

    def test_rejects_nested_optimizer_args_values(self) -> None:
        from northstar.config.loader import load_run_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid-optimizer-args.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    name = "flux2-smoke"
                    architecture = "flux2"

                    [model]
                    version = "klein-base-9b"
                    dit = "/models/flux2.safetensors"

                    [optimizer]
                    type = "adafactor"

                    [optimizer.args]
                    scale_parameter = false

                    [optimizer.args.nested]
                    bad = true
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "optimizer.args.nested"):
                load_run_config(config_path)

    def test_optimizer_args_type_matches_validated_shape(self) -> None:
        from northstar.config.models import OptimizerConfigSection

        args_type = typing.get_type_hints(OptimizerConfigSection)["args"]

        self.assertEqual(str(args_type), "dict[str, bool | int | float | str] | None")


if __name__ == "__main__":
    unittest.main()
