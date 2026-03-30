import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from musubi_tuner.ltx2_prompt_lora_utils import resolve_ltx_prompt_file_data
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer


class LTX2PromptLoraUtilsTests(unittest.TestCase):
    def test_root_loras_are_applied_as_baseline(self):
        data = {
            "prompt": {
                "width": 1280,
                "height": 832,
                "loras": [{"path": "/base.safetensors", "weight": 0.3, "merge": False}],
                "subset": [{"prompt": "baseline prompt"}],
            }
        }

        prompts = resolve_ltx_prompt_file_data(data)

        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]["width"], 1280)
        self.assertEqual(prompts[0]["height"], 832)
        self.assertEqual(prompts[0]["enum"], 0)
        self.assertEqual(
            prompts[0]["resolved_loras"],
            [{"path": "/base.safetensors", "weight": 0.3, "merge": False}],
        )

    def test_subset_loras_default_to_extend_mode(self):
        data = {
            "prompt": {
                "loras": [{"path": "/base.safetensors", "weight": 0.3, "merge": False}],
                "subset": [
                    {
                        "prompt": "extend prompt",
                        "loras": [{"path": "/extra.safetensors", "weight": 0.8, "merge": False}],
                    }
                ],
            }
        }

        prompts = resolve_ltx_prompt_file_data(data)

        self.assertEqual(
            prompts[0]["resolved_loras"],
            [
                {"path": "/base.safetensors", "weight": 0.3, "merge": False},
                {"path": "/extra.safetensors", "weight": 0.8, "merge": False},
            ],
        )

    def test_subset_replace_mode_drops_root_baseline(self):
        data = {
            "prompt": {
                "loras": [{"path": "/base.safetensors", "weight": 0.3, "merge": False}],
                "subset": [
                    {
                        "prompt": "replace prompt",
                        "lora_mode": "replace",
                        "loras": [{"path": "/replace.safetensors", "weight": 0.9, "merge": False}],
                    }
                ],
            }
        }

        prompts = resolve_ltx_prompt_file_data(data)

        self.assertEqual(
            prompts[0]["resolved_loras"],
            [{"path": "/replace.safetensors", "weight": 0.9, "merge": False}],
        )

    def test_explicit_extend_mode_keeps_root_baseline(self):
        data = {
            "prompt": {
                "loras": [{"path": "/base.safetensors", "weight": 0.3, "merge": False}],
                "subset": [
                    {
                        "prompt": "extend prompt",
                        "lora_mode": "extend",
                        "loras": [{"path": "/extra.safetensors", "weight": 0.8, "merge": False}],
                    }
                ],
            }
        }

        prompts = resolve_ltx_prompt_file_data(data)

        self.assertEqual(
            prompts[0]["resolved_loras"],
            [
                {"path": "/base.safetensors", "weight": 0.3, "merge": False},
                {"path": "/extra.safetensors", "weight": 0.8, "merge": False},
            ],
        )

    def test_invalid_lora_mode_raises_value_error(self):
        data = {
            "prompt": {
                "subset": [
                    {
                        "prompt": "bad mode",
                        "lora_mode": "invalid",
                        "loras": [{"path": "/bad.safetensors"}],
                    }
                ]
            }
        }

        with self.assertRaises(ValueError):
            resolve_ltx_prompt_file_data(data)

    def test_missing_lora_path_raises_value_error(self):
        data = {
            "prompt": {
                "subset": [
                    {
                        "prompt": "missing path",
                        "loras": [{"weight": 0.5, "merge": False}],
                    }
                ]
            }
        }

        with self.assertRaises(ValueError):
            resolve_ltx_prompt_file_data(data)

    def test_invalid_loras_shape_raises_value_error(self):
        data = {
            "prompt": {
                "subset": [
                    {
                        "prompt": "bad loras",
                        "loras": "not-a-list",
                    }
                ]
            }
        }

        with self.assertRaises(ValueError):
            resolve_ltx_prompt_file_data(data)

    def test_invalid_weight_raises_value_error(self):
        data = {
            "prompt": {
                "subset": [
                    {
                        "prompt": "bad weight",
                        "loras": [{"path": "/bad.safetensors", "weight": "oops"}],
                    }
                ]
            }
        }

        with self.assertRaises(ValueError):
            resolve_ltx_prompt_file_data(data)

    def test_process_sample_prompts_resolves_loras_for_toml_prompt_file(self):
        trainer = LTX2NetworkTrainer()
        args = Namespace(
            use_precached_sample_prompts=False,
            precache_sample_prompts=False,
            use_precached_sample_latents=False,
            height=512,
            width=768,
            sample_num_frames=45,
            guidance_scale=3.0,
            discrete_flow_shift=5.0,
            sample_sigmas=None,
            sample_stage1_distilled_lora_multiplier=1.0,
            sample_stage2_distilled_lora_multiplier=1.0,
            lora_weight=["/cli_base.safetensors"],
            lora_multiplier=[0.4],
        )
        toml_text = """
[prompt]
width = 1280
height = 832
loras = [
  { path = "/root_base.safetensors", weight = 0.3, merge = false }
]

[[prompt.subset]]
prompt = "Prompt A"
loras = [
  { path = "/subset_a.safetensors", weight = 0.8, merge = false }
]

[[prompt.subset]]
prompt = "Prompt B"
lora_mode = "replace"
loras = [
  { path = "/subset_b.safetensors", weight = 0.9, merge = false }
]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "prompts.toml"
            prompt_file.write_text(toml_text, encoding="utf-8")

            sample_params = trainer.process_sample_prompts(args, accelerator=None, sample_prompts=str(prompt_file))

        self.assertEqual(len(sample_params), 2)
        self.assertEqual(sample_params[0]["width"], 1280)
        self.assertEqual(sample_params[0]["height"], 832)
        self.assertEqual(sample_params[0]["enum"], 0)
        self.assertEqual(
            sample_params[0]["resolved_loras"],
            [
                {"path": "/cli_base.safetensors", "weight": 0.4, "merge": False},
                {"path": "/root_base.safetensors", "weight": 0.3, "merge": False},
                {"path": "/subset_a.safetensors", "weight": 0.8, "merge": False},
            ],
        )
        self.assertEqual(
            sample_params[1]["resolved_loras"],
            [{"path": "/subset_b.safetensors", "weight": 0.9, "merge": False}],
        )


if __name__ == "__main__":
    unittest.main()
