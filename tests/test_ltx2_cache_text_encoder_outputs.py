import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from musubi_tuner import ltx2_cache_text_encoder_outputs as cache_script


class LTX2CacheTextEncoderOutputsTests(unittest.TestCase):
    def test_load_sample_prompts_for_precache_uses_ltx_toml_resolution(self):
        args = Namespace(
            sample_prompts=None,
            lora_weight=["/cli_base.safetensors"],
            lora_multiplier=[0.4],
        )
        toml_text = """
[prompt]
negative_prompt = "root negative"
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
            args.sample_prompts = str(prompt_file)

            prompts = cache_script._load_sample_prompts_for_precache(args)

        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0]["negative_prompt"], "root negative")
        self.assertEqual(
            prompts[0]["resolved_loras"],
            [
                {"path": "/cli_base.safetensors", "weight": 0.4, "merge": False},
                {"path": "/root_base.safetensors", "weight": 0.3, "merge": False},
                {"path": "/subset_a.safetensors", "weight": 0.8, "merge": False},
            ],
        )
        self.assertEqual(
            prompts[1]["resolved_loras"],
            [{"path": "/subset_b.safetensors", "weight": 0.9, "merge": False}],
        )

    def test_resolve_sample_prompts_cache_path_prefers_explicit_path_without_datasets(self):
        args = Namespace(sample_prompts_cache="/tmp/sample-prompts-cache.pt")

        cache_path = cache_script._resolve_sample_prompts_cache_path(args, datasets=[])

        self.assertEqual(cache_path, "/tmp/sample-prompts-cache.pt")

    def test_should_cache_sample_prompts_only_reads_flag(self):
        self.assertTrue(cache_script._should_cache_sample_prompts_only(Namespace(cache_sample_prompts_only=True)))
        self.assertFalse(cache_script._should_cache_sample_prompts_only(Namespace(cache_sample_prompts_only=False)))


if __name__ == "__main__":
    unittest.main()
