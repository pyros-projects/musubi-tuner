import unittest

import torch
import torch.nn as nn

from musubi_tuner.hv_train_network import NetworkTrainer
from musubi_tuner.networks import lora


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.lin(x)


class _ToyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_ToyBlock()])

    def forward(self, x):
        return self.blocks[0](x)


class SamplingAdapterCompileTest(unittest.TestCase):
    def test_compiled_blocks_are_discovered_without_eager_swap(self):
        transformer = _ToyTransformer()
        transformer.blocks[0] = torch.compile(transformer.blocks[0], backend="eager")

        weights_sd = {
            "lora_unet_blocks_0_lin.lora_down.weight": torch.ones(2, 4),
            "lora_unet_blocks_0_lin.lora_up.weight": torch.ones(4, 2),
            "lora_unet_blocks_0_lin.alpha": torch.tensor(2.0),
        }

        compiled_network = lora.create_network_from_weights(None, 1.0, weights_sd, None, transformer, True)
        self.assertEqual([module.lora_name for module in compiled_network.unet_loras], ["lora_unet_blocks_0_lin"])

    def test_sampling_patch_and_restore_work_on_compiled_module(self):
        transformer = _ToyTransformer()
        with torch.no_grad():
            transformer.blocks[0].lin.weight.copy_(torch.eye(4))
        transformer.blocks[0] = torch.compile(transformer.blocks[0], backend="eager")

        weights_sd = {
            "lora_unet_blocks_0_lin.lora_down.weight": torch.ones(2, 4),
            "lora_unet_blocks_0_lin.lora_up.weight": torch.ones(4, 2),
            "lora_unet_blocks_0_lin.alpha": torch.tensor(2.0),
        }
        network = lora.create_network_from_weights(None, 1.0, weights_sd, None, transformer, True)
        trainer = NetworkTrainer()
        x = torch.ones(1, 4)

        baseline = transformer(x)
        patched, runtime_attached, backups = trainer._apply_sampling_lora_network(network, weights_sd, torch.device("cpu"))
        patched_output = transformer(x)
        restored = trainer._restore_sampling_lora_network(backups)
        restored_output = transformer(x)

        self.assertEqual(patched, 1)
        self.assertEqual(runtime_attached, 0)
        self.assertEqual(restored, 1)
        self.assertFalse(torch.allclose(patched_output, baseline))
        self.assertTrue(torch.allclose(restored_output, baseline))
        self.assertEqual(transformer.blocks[0].__class__.__name__, "OptimizedModule")

    def test_float8_like_sampling_uses_runtime_overlay(self):
        transformer = _ToyTransformer()
        transformer = transformer.to(dtype=torch.float16)
        weights_sd = {
            "lora_unet_blocks_0_lin.lora_down.weight": torch.ones(2, 4),
            "lora_unet_blocks_0_lin.lora_up.weight": torch.ones(4, 2),
            "lora_unet_blocks_0_lin.alpha": torch.tensor(2.0),
        }
        network = lora.create_network_from_weights(None, 1.0, weights_sd, None, transformer, True)
        trainer = NetworkTrainer()
        trainer._sampling_lora_float8_dtypes = lambda: (torch.float16,)
        x = torch.ones(1, 4, dtype=torch.float16)

        baseline = transformer(x)
        merged, runtime_attached, backups = trainer._apply_sampling_lora_network(network, weights_sd, torch.device("cpu"))
        patched_output = transformer(x)
        cleared = trainer._clear_sampling_lora_runtime_overlays(transformer)
        restored_output = transformer(x)

        self.assertEqual(merged, 0)
        self.assertEqual(runtime_attached, 1)
        self.assertEqual(len(backups), 0)
        self.assertEqual(cleared, 1)
        self.assertFalse(torch.allclose(patched_output, baseline))
        self.assertTrue(torch.allclose(restored_output, baseline))

    def test_compiled_float8_like_sampling_requests_eager_fallback(self):
        transformer = _ToyTransformer().to(dtype=torch.float16)
        transformer.blocks[0] = torch.compile(transformer.blocks[0], backend="eager")

        trainer = NetworkTrainer()
        trainer._sampling_lora_float8_dtypes = lambda: (torch.float16,)

        self.assertTrue(trainer._sampling_lora_requires_eager_fallback(transformer))


if __name__ == "__main__":
    unittest.main()
