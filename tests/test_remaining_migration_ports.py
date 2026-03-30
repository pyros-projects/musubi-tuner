from __future__ import annotations

import importlib
import sys
import types
import unittest
import tempfile
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from musubi_tuner.hv_train_network import NetworkTrainer, enable_gradient_checkpointing_compat
from musubi_tuner.hv_train import FineTuningTrainer
from musubi_tuner.networks import lora_flux_2
from musubi_tuner.zimage_train_network import ZImageNetworkTrainer


class _ToyFlux2(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_in = nn.Linear(4, 4, bias=False)


class _CompatTransformerNoBlocks:
    def __init__(self):
        self.calls = []

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.calls.append({"cpu_offload": cpu_offload})


class _CompatTransformerWithBlocks:
    def __init__(self):
        self.calls = []

    def enable_gradient_checkpointing(
        self,
        cpu_offload: bool = False,
        weight_cpu_offloading: bool = False,
        blocks_to_checkpoint: int = -1,
    ):
        self.calls.append(
            {
                "cpu_offload": cpu_offload,
                "weight_cpu_offloading": weight_cpu_offloading,
                "blocks_to_checkpoint": blocks_to_checkpoint,
            }
        )


class _FakeAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return mock.MagicMock(__enter__=lambda *_: None, __exit__=lambda *_: False)


class _FakeSamplingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.inference_switches = 0
        self.training_switches = 0

    def switch_block_swap_for_inference(self):
        self.inference_switches += 1

    def switch_block_swap_for_training(self):
        self.training_switches += 1


class _FakeFloat8Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.zeros((4, 4), dtype=torch.float8_e4m3fn))

    def forward(self, x):
        return x


class _FakeSamplingLora:
    def __init__(self, module: nn.Module, name: str = "lora_unet_test"):
        self.org_module_ref = [module]
        self.split_dims = None
        self.multiplier = 1.0
        self.scale = 1.0
        self.lora_name = name


class _SamplingCompatTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.events = []

    def _get_sampling_lora_specs(self, args):
        return [("dummy.safetensors", 0.8)]

    def _sampling_lora_requires_eager_fallback(self, transformer):
        return False

    def _apply_sampling_lora(self, args, transformer, device):
        self.events.append(("apply", device.type))
        return [("dummy.safetensors", {})]

    def _restore_sampling_lora(self, args, transformer, device, applied):
        self.events.append(("restore", len(applied), device.type))

    def sample_image_inference(self, accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps):
        self.events.append(("sample", sample_parameter["prompt"]))


class RemainingMigrationPortsTest(unittest.TestCase):
    def test_prodigyplus_optimizer_package_is_available_from_repo_src(self) -> None:
        module = importlib.import_module("prodigyplus.prodigy_plus_schedulefree")
        module_path = Path(module.__file__).resolve()
        repo_src = Path(__file__).resolve().parents[1] / "src"

        self.assertTrue(module_path.is_relative_to(repo_src), module_path)
        self.assertTrue(hasattr(module, "ProdigyPlusScheduleFree"))

    def test_shared_sample_images_applies_and_restores_sampling_lora(self) -> None:
        trainer = _SamplingCompatTrainer()
        accelerator = _FakeAccelerator()
        transformer = _FakeSamplingTransformer()
        args = types.SimpleNamespace(
            sample_every_n_steps=1,
            sample_every_n_epochs=None,
            sample_at_first=False,
            output_dir=tempfile.mkdtemp(),
            sample_prompts="dummy.txt",
            compile=False,
            sampling_lora_weight=["dummy.safetensors"],
            sampling_lora_multiplier=[0.8],
        )

        trainer.sample_images(
            accelerator,
            args,
            epoch=None,
            steps=1,
            vae=None,
            transformer=transformer,
            sample_parameters=[{"prompt": "hello"}],
            dit_dtype=torch.float32,
        )

        self.assertEqual(
            trainer.events,
            [("apply", "cpu"), ("sample", "hello"), ("restore", 1, "cpu")],
        )

    def test_sampling_lora_runtime_overlay_uses_float32_on_float8_modules(self) -> None:
        trainer = NetworkTrainer()
        module = _FakeFloat8Linear()
        lora = _FakeSamplingLora(module)
        network = types.SimpleNamespace(text_encoder_loras=[], unet_loras=[lora])
        weights_sd = {
            "lora_unet_test.lora_down.weight": torch.ones((2, 4), dtype=torch.float16),
            "lora_unet_test.lora_up.weight": torch.ones((4, 2), dtype=torch.float16),
        }

        merged, runtime_attached, backups = trainer._apply_sampling_lora_network(network, weights_sd, torch.device("cpu"))

        self.assertEqual(merged, 0)
        self.assertEqual(runtime_attached, 1)
        self.assertEqual(backups, {})
        self.assertEqual(getattr(module, "_sampling_lora_runtime_A_0").dtype, torch.float32)
        self.assertEqual(getattr(module, "_sampling_lora_runtime_B_0").dtype, torch.float32)

    def test_gc_compat_omits_unsupported_block_kwargs(self) -> None:
        transformer = _CompatTransformerNoBlocks()

        enable_gradient_checkpointing_compat(
            transformer,
            cpu_offload=True,
            weight_cpu_offloading=False,
            blocks_to_checkpoint=7,
        )

        self.assertEqual(transformer.calls, [{"cpu_offload": True}])

    def test_gc_compat_passes_block_kwargs_when_supported(self) -> None:
        transformer = _CompatTransformerWithBlocks()

        enable_gradient_checkpointing_compat(
            transformer,
            cpu_offload=True,
            weight_cpu_offloading=True,
            blocks_to_checkpoint=7,
        )

        self.assertEqual(
            transformer.calls,
            [{"cpu_offload": True, "weight_cpu_offloading": True, "blocks_to_checkpoint": 7}],
        )

    def test_flux2_loader_detects_non_block_linear_targets(self) -> None:
        transformer = _ToyFlux2()
        weights_sd = {
            "lora_unet_img_in.lora_down.weight": torch.ones(2, 4),
            "lora_unet_img_in.lora_up.weight": torch.ones(4, 2),
            "lora_unet_img_in.alpha": torch.tensor(2.0),
        }

        network = lora_flux_2.create_arch_network_from_weights(
            1.0,
            weights_sd,
            text_encoders=None,
            unet=transformer,
            for_inference=True,
        )

        self.assertEqual([module.lora_name for module in network.unet_loras], ["lora_unet_img_in"])

    def test_zimage_sampling_lora_normalization_splits_qkv_and_adds_missing_alpha(self) -> None:
        trainer = ZImageNetworkTrainer()
        args = types.SimpleNamespace(network_module="networks.lora_zimage")
        weights_sd = {
            "lora_unet_transformer_blocks_0_attention_qkv.lora_down.weight": torch.ones(2, 4),
            "lora_unet_transformer_blocks_0_attention_qkv.lora_up.weight": torch.arange(24, dtype=torch.float32).reshape(12, 2),
        }

        normalized = trainer.normalize_sampling_lora_weights(args, weights_sd, "dummy.safetensors")

        expected_prefixes = [
            "lora_unet_transformer_blocks_0_attention_to_q",
            "lora_unet_transformer_blocks_0_attention_to_k",
            "lora_unet_transformer_blocks_0_attention_to_v",
        ]
        self.assertEqual(
            {key.rsplit(".", 2)[0] for key in normalized if key.endswith(".lora_up.weight")},
            set(expected_prefixes),
        )
        for prefix in expected_prefixes:
            self.assertIn(f"{prefix}.lora_down.weight", normalized)
            self.assertIn(f"{prefix}.lora_up.weight", normalized)
            self.assertIn(f"{prefix}.alpha", normalized)
            self.assertEqual(normalized[f"{prefix}.alpha"].item(), 2.0)

        self.assertNotIn("lora_unet_transformer_blocks_0_attention_qkv.lora_up.weight", normalized)

    def test_hv_trainer_supports_prodigy_plus_schedule_free_optimizer(self) -> None:
        trainer = FineTuningTrainer()
        params = [nn.Parameter(torch.ones(1))]
        args = types.SimpleNamespace(
            optimizer_type="ProdigyPlusScheduleFree",
            optimizer_args=["betas=(0.9, 0.99)", "weight_decay=0.0"],
            learning_rate=1.0,
            lr_scheduler="constant",
            max_grad_norm=0.0,
        )

        fake_pkg = types.ModuleType("prodigyplus")
        fake_submodule = types.ModuleType("prodigyplus.prodigy_plus_schedulefree")

        class ProdigyPlusScheduleFree(torch.optim.Optimizer):
            def __init__(self, params, lr=1.0, **kwargs):
                defaults = {"lr": lr, **kwargs}
                super().__init__(params, defaults)

            def step(self, closure=None):
                return None

        ProdigyPlusScheduleFree.__module__ = "prodigyplus.prodigy_plus_schedulefree"
        fake_submodule.ProdigyPlusScheduleFree = ProdigyPlusScheduleFree
        fake_pkg.prodigy_plus_schedulefree = fake_submodule

        with mock.patch.dict(
            sys.modules,
            {
                "prodigyplus": fake_pkg,
                "prodigyplus.prodigy_plus_schedulefree": fake_submodule,
            },
        ):
            optimizer_name, optimizer_args, optimizer, _train_fn, _eval_fn = trainer.get_optimizer(args, params)

        self.assertEqual(optimizer_name, "prodigyplus.prodigy_plus_schedulefree.ProdigyPlusScheduleFree")
        self.assertEqual(optimizer.__class__.__name__, "ProdigyPlusScheduleFree")
        self.assertIn("betas=(0.9, 0.99)", optimizer_args)

    def test_hv_network_trainer_supports_prodigy_plus_schedule_free_optimizer(self) -> None:
        trainer = NetworkTrainer()
        params = [nn.Parameter(torch.ones(1))]
        args = types.SimpleNamespace(
            optimizer_type="ProdigyPlusScheduleFree",
            optimizer_args=["betas=(0.9, 0.99)", "weight_decay=0.0"],
            learning_rate=1.0,
            lr_scheduler="constant",
            max_grad_norm=0.0,
        )

        fake_pkg = types.ModuleType("prodigyplus")
        fake_submodule = types.ModuleType("prodigyplus.prodigy_plus_schedulefree")

        class ProdigyPlusScheduleFree(torch.optim.Optimizer):
            def __init__(self, params, lr=1.0, **kwargs):
                defaults = {"lr": lr, **kwargs}
                super().__init__(params, defaults)

            def step(self, closure=None):
                return None

        ProdigyPlusScheduleFree.__module__ = "prodigyplus.prodigy_plus_schedulefree"
        fake_submodule.ProdigyPlusScheduleFree = ProdigyPlusScheduleFree
        fake_pkg.prodigy_plus_schedulefree = fake_submodule

        with mock.patch.dict(
            sys.modules,
            {
                "prodigyplus": fake_pkg,
                "prodigyplus.prodigy_plus_schedulefree": fake_submodule,
            },
        ):
            optimizer_name, optimizer_args, optimizer, _train_fn, _eval_fn = trainer.get_optimizer(args, params)

        self.assertEqual(optimizer_name, "prodigyplus.prodigy_plus_schedulefree.ProdigyPlusScheduleFree")
        self.assertEqual(optimizer.__class__.__name__, "ProdigyPlusScheduleFree")
        self.assertIn("betas=(0.9, 0.99)", optimizer_args)


if __name__ == "__main__":
    unittest.main()
