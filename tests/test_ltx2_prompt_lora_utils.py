import base64
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from musubi_tuner.ltx2_prompt_lora_utils import resolve_ltx_prompt_file_data
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer


class LTX2PromptLoraUtilsTests(unittest.TestCase):
    @staticmethod
    def _write_png(path: Path) -> str:
        path.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
            )
        )
        return str(path)

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

    def test_unique_image_pool_assigns_distinct_image_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_dir = Path(tmpdir) / "pool"
            pool_dir.mkdir()
            image_a = self._write_png(pool_dir / "a.png")
            image_b = self._write_png(pool_dir / "b.png")

            data = {
                "prompt": {
                    "input_images_dir": str(pool_dir),
                    "image_assignment": "unique",
                    "subset": [
                        {"prompt": "subset one"},
                        {"prompt": "subset two"},
                    ],
                }
            }

            prompts = resolve_ltx_prompt_file_data(data)

        self.assertCountEqual([prompt.get("image_path") for prompt in prompts], [image_a, image_b])

    def test_random_image_input_order_uses_shuffled_pool_assignment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_dir = Path(tmpdir) / "pool"
            pool_dir.mkdir()
            image_1 = self._write_png(pool_dir / "1.png")
            image_2 = self._write_png(pool_dir / "2.png")
            image_3 = self._write_png(pool_dir / "3.png")

            data = {
                "prompt": {
                    "input_images_dir": str(pool_dir),
                    "image_assignment": "unique",
                    "image_input_order": "random",
                    "subset": [
                        {"prompt": "subset one"},
                        {"prompt": "subset two"},
                        {"prompt": "subset three"},
                    ],
                }
            }

            with mock.patch("random.shuffle", side_effect=lambda seq: seq.reverse()) as shuffle_mock:
                prompts = resolve_ltx_prompt_file_data(data)

        self.assertEqual(
            [prompt.get("image_path") for prompt in prompts],
            [image_3, image_2, image_1],
        )
        shuffle_mock.assert_called_once()

    def test_explicit_image_path_takes_precedence_over_pool_assignment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pool_dir = tmpdir_path / "pool"
            pool_dir.mkdir()
            pool_image_1 = self._write_png(pool_dir / "pool_1.png")
            pool_image_2 = self._write_png(pool_dir / "pool_2.png")
            explicit_image = self._write_png(tmpdir_path / "explicit.png")

            data = {
                "prompt": {
                    "input_images_dir": str(pool_dir),
                    "image_assignment": "unique",
                    "subset": [
                        {"prompt": "explicit subset", "image_path": explicit_image},
                        {"prompt": "pool subset"},
                    ],
                }
            }

            prompts = resolve_ltx_prompt_file_data(data)

        self.assertEqual(prompts[0].get("image_path"), explicit_image)
        self.assertIn(prompts[1].get("image_path"), {pool_image_1, pool_image_2})

    def test_use_image_pool_false_leaves_image_path_unset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_dir = Path(tmpdir) / "pool"
            pool_dir.mkdir()
            pool_image = self._write_png(pool_dir / "pool.png")

            data = {
                "prompt": {
                    "input_images_dir": str(pool_dir),
                    "subset": [
                        {"prompt": "t2v subset", "use_image_pool": False},
                        {"prompt": "pool subset"},
                    ],
                }
            }

            prompts = resolve_ltx_prompt_file_data(data)

        self.assertNotIn("image_path", prompts[0])
        self.assertEqual(prompts[1].get("image_path"), pool_image)

    def test_mixed_prompt_file_supports_pool_explicit_and_t2v_subsets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pool_dir = tmpdir_path / "pool"
            pool_dir.mkdir()
            pool_image_1 = self._write_png(pool_dir / "pool_1.png")
            pool_image_2 = self._write_png(pool_dir / "pool_2.png")
            explicit_image = self._write_png(tmpdir_path / "explicit.png")

            data = {
                "prompt": {
                    "input_images_dir": str(pool_dir),
                    "image_assignment": "unique",
                    "subset": [
                        {"prompt": "pool subset"},
                        {"prompt": "explicit subset", "image_path": explicit_image},
                        {"prompt": "t2v subset", "use_image_pool": False},
                    ],
                }
            }

            prompts = resolve_ltx_prompt_file_data(data)

        self.assertEqual(prompts[0].get("image_path"), pool_image_1)
        self.assertEqual(prompts[1].get("image_path"), explicit_image)
        self.assertNotIn("image_path", prompts[2])

    def test_invalid_image_input_order_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_dir = Path(tmpdir) / "pool"
            pool_dir.mkdir()
            self._write_png(pool_dir / "image.png")

            data = {
                "prompt": {
                    "input_images_dir": str(pool_dir),
                    "image_input_order": "sideways",
                    "subset": [{"prompt": "pool subset"}],
                }
            }

            with self.assertRaises(ValueError):
                resolve_ltx_prompt_file_data(data)

    def test_invalid_image_assignment_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_dir = Path(tmpdir) / "pool"
            pool_dir.mkdir()
            self._write_png(pool_dir / "image.png")

            data = {
                "prompt": {
                    "input_images_dir": str(pool_dir),
                    "image_assignment": "invalid",
                    "subset": [{"prompt": "pool subset"}],
                }
            }

            with self.assertRaises(ValueError):
                resolve_ltx_prompt_file_data(data)

    def test_nonexistent_input_images_dir_raises_file_not_found_error(self):
        data = {
            "prompt": {
                "input_images_dir": "/no/such/image-pool",
                "subset": [{"prompt": "pool subset"}],
            }
        }

        with self.assertRaises(FileNotFoundError):
            resolve_ltx_prompt_file_data(data)

    def test_nonexistent_explicit_image_path_raises_file_not_found_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_dir = Path(tmpdir) / "pool"
            pool_dir.mkdir()
            self._write_png(pool_dir / "pool.png")

            data = {
                "prompt": {
                    "input_images_dir": str(pool_dir),
                    "subset": [
                        {
                            "prompt": "explicit subset",
                            "image_path": str(Path(tmpdir) / "missing.png"),
                        }
                    ],
                }
            }

            with self.assertRaises(FileNotFoundError):
                resolve_ltx_prompt_file_data(data)

    def test_unique_image_assignment_requires_enough_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_dir = Path(tmpdir) / "pool"
            pool_dir.mkdir()
            self._write_png(pool_dir / "only.png")

            data = {
                "prompt": {
                    "input_images_dir": str(pool_dir),
                    "image_assignment": "unique",
                    "subset": [
                        {"prompt": "subset one"},
                        {"prompt": "subset two"},
                    ],
                }
            }

            with self.assertRaises(ValueError):
                resolve_ltx_prompt_file_data(data)

    def test_empty_supported_image_pool_errors_when_subset_needs_pool_assignment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_dir = Path(tmpdir) / "pool"
            pool_dir.mkdir()
            (pool_dir / "notes.txt").write_text("not an image", encoding="utf-8")

            data = {
                "prompt": {
                    "input_images_dir": str(pool_dir),
                    "subset": [{"prompt": "pool subset"}],
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
