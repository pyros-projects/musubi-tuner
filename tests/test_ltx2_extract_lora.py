import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from musubi_tuner.ltx2_extract_lora import TensorSource


class LTX2ExtractLoraTests(unittest.TestCase):
    @unittest.skipIf(not hasattr(torch, "float8_e4m3fn"), "float8 dtype unavailable")
    def test_tensor_source_dequantizes_scaled_fp8_weight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scaled_fp8.safetensors"
            key = "model.diffusion_model.transformer_blocks.10.attn1.to_k.weight"
            weight = torch.tensor([[1.0, -2.0], [4.0, -8.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
            scale = torch.tensor(0.25, dtype=torch.float32)
            save_file(
                {
                    key: weight,
                    key.replace(".weight", ".weight_scale"): scale,
                },
                path,
            )

            tensor = TensorSource(path).get_tensor(key)

            self.assertEqual(tensor.dtype, torch.float32)
            torch.testing.assert_close(tensor, weight.to(torch.float32) * scale)

    @unittest.skipIf(not hasattr(torch, "float8_e4m3fn"), "float8 dtype unavailable")
    def test_tensor_source_dequantizes_block_scaled_fp8_weight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "block_scaled_fp8.safetensors"
            key = "model.diffusion_model.transformer_blocks.10.attn1.to_q.weight"
            weight = torch.tensor(
                [[1.0, -2.0, 4.0, -8.0], [16.0, -32.0, 64.0, -128.0]],
                dtype=torch.float32,
            ).to(torch.float8_e4m3fn)
            scale = torch.tensor([[[0.5], [0.25]], [[0.125], [0.0625]]], dtype=torch.float32)
            save_file(
                {
                    key: weight,
                    key.replace(".weight", ".weight_scale"): scale,
                },
                path,
            )

            tensor = TensorSource(path).get_tensor(key)
            expected = weight.to(torch.float32).view(2, 2, 2) * scale

            self.assertEqual(tensor.dtype, torch.float32)
            torch.testing.assert_close(tensor, expected.view(weight.shape))


if __name__ == "__main__":
    unittest.main()
