import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from musubi_tuner.boogu_image_extract_diff_lora import (
    TensorSource,
    build_extraction_groups,
    extract_diff_lora,
)


class BooguImageExtractDiffLoraTests(unittest.TestCase):
    def test_build_groups_uses_boogu_comfy_diff_key_conventions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.safetensors"
            target_path = Path(tmpdir) / "target.safetensors"
            base_state = {
                "block.attn.to_q.weight": torch.zeros(2, 2),
                "block.norm.weight": torch.zeros(2),
                "block.linear.bias": torch.zeros(2),
                "image_index_embedding": torch.zeros(1, 2),
            }
            target_state = {
                **base_state,
                "missing_in_base.weight": torch.ones(2, 2),
            }
            save_file(base_state, base_path)
            save_file(target_state, target_path)

            groups = build_extraction_groups(TensorSource(target_path), TensorSource(base_path))

            self.assertEqual([group.source_key for group in groups.lora], ["block.attn.to_q.weight"])
            self.assertEqual(groups.lora[0].down_key, "diffusion_model.block.attn.to_q.lora_down.weight")
            self.assertEqual(groups.lora[0].up_key, "diffusion_model.block.attn.to_q.lora_up.weight")
            self.assertEqual(
                [group.output_key for group in groups.direct],
                ["diffusion_model.block.linear.diff_b", "diffusion_model.block.norm.diff"],
            )
            self.assertEqual([item["key"] for item in groups.skipped], ["image_index_embedding"])

    def test_extract_diff_lora_writes_lora_and_direct_diff_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.safetensors"
            target_path = Path(tmpdir) / "target.safetensors"
            out_path = Path(tmpdir) / "diff_lora.safetensors"
            report_path = Path(tmpdir) / "report.json"
            base_state = {
                "block.attn.to_q.weight": torch.zeros(2, 2),
                "block.norm.weight": torch.ones(2),
                "block.linear.bias": torch.tensor([3.0, 4.0]),
                "image_index_embedding": torch.zeros(1, 2),
            }
            target_state = {
                "block.attn.to_q.weight": torch.tensor([[2.0, 0.0], [0.0, 0.0]]),
                "block.norm.weight": torch.tensor([1.5, 0.25]),
                "block.linear.bias": torch.tensor([2.5, 6.0]),
                "image_index_embedding": torch.ones(1, 2),
            }
            save_file(base_state, base_path)
            save_file(target_state, target_path)

            summary = extract_diff_lora(
                model_path=target_path,
                base_model_path=base_path,
                out_path=out_path,
                report_path=report_path,
                max_rank=1,
                device=torch.device("cpu"),
                save_dtype=torch.float32,
            )

            weights = load_file(out_path)
            down = weights["diffusion_model.block.attn.to_q.lora_down.weight"]
            up = weights["diffusion_model.block.attn.to_q.lora_up.weight"]
            torch.testing.assert_close(up @ down, target_state["block.attn.to_q.weight"])
            torch.testing.assert_close(weights["diffusion_model.block.norm.diff"], torch.tensor([0.5, -0.75]))
            torch.testing.assert_close(weights["diffusion_model.block.linear.diff_b"], torch.tensor([-0.5, 2.0]))
            self.assertEqual(summary["output_tensors"], 4)
            self.assertEqual(summary["extracted_lora_groups"], 1)
            self.assertEqual(summary["direct_diff_groups"], 2)
            self.assertEqual(summary["skipped_groups"], 1)
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
