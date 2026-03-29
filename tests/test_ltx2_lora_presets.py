import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musubi_tuner.ltx2_train_network import ltx2_setup_parser
from musubi_tuner.networks.lora_ltx2 import _get_include_patterns_for_preset


class LTX2LoraPresetTests(unittest.TestCase):
    def test_video_cross_attn_preset_matches_video_text_cross_attention_only(self):
        self.assertEqual(
            _get_include_patterns_for_preset("video_cross_attn"),
            [
                r".*\.attn2\.to_k$",
                r".*\.attn2\.to_q$",
                r".*\.attn2\.to_v$",
                r".*\.attn2\.to_out\.0$",
            ],
        )

    def test_video_cross_attn_ffn_preset_adds_video_feed_forward_layers(self):
        self.assertEqual(
            _get_include_patterns_for_preset("video_cross_attn_ffn"),
            [
                r".*\.attn2\.to_k$",
                r".*\.attn2\.to_q$",
                r".*\.attn2\.to_v$",
                r".*\.attn2\.to_out\.0$",
                r".*\.ff\.net\.0\.proj$",
                r".*\.ff\.net\.2$",
            ],
        )

    def test_video_attn_preset_matches_video_attention_only(self):
        self.assertEqual(
            _get_include_patterns_for_preset("video_attn"),
            [
                r".*\.attn1\.to_k$",
                r".*\.attn1\.to_q$",
                r".*\.attn1\.to_v$",
                r".*\.attn1\.to_out\.0$",
                r".*\.attn2\.to_k$",
                r".*\.attn2\.to_q$",
                r".*\.attn2\.to_v$",
                r".*\.attn2\.to_out\.0$",
            ],
        )

    def test_video_attn_ffn_preset_adds_video_feed_forward_layers(self):
        self.assertEqual(
            _get_include_patterns_for_preset("video_attn_ffn"),
            [
                r".*\.attn1\.to_k$",
                r".*\.attn1\.to_q$",
                r".*\.attn1\.to_v$",
                r".*\.attn1\.to_out\.0$",
                r".*\.attn2\.to_k$",
                r".*\.attn2\.to_q$",
                r".*\.attn2\.to_v$",
                r".*\.attn2\.to_out\.0$",
                r".*\.ff\.net\.0\.proj$",
                r".*\.ff\.net\.2$",
            ],
        )

    def test_parser_accepts_new_lora_target_presets(self):
        parser = ltx2_setup_parser(argparse.ArgumentParser())
        action = next(a for a in parser._actions if a.dest == "lora_target_preset")

        self.assertIn("video_attn", action.choices)
        self.assertIn("video_attn_ffn", action.choices)
        self.assertIn("video_cross_attn", action.choices)
        self.assertIn("video_cross_attn_ffn", action.choices)


if __name__ == "__main__":
    unittest.main()
