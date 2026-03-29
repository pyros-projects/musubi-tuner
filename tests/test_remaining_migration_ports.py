from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

from musubi_tuner.hv_train import FineTuningTrainer
from musubi_tuner.networks import lora_flux_2
from musubi_tuner.zimage_train_network import ZImageNetworkTrainer


class _ToyFlux2(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_in = nn.Linear(4, 4, bias=False)


class RemainingMigrationPortsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
