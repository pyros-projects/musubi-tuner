from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import torch

from musubi_tuner import zimage_generate_image


class _DummyZImageModel:
    def __init__(self):
        self.noise_refiner = object()
        self.context_refiner = object()
        self.layers = object()

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self

    def requires_grad_(self, _flag):
        return self


class ZImageGenerateImageTest(unittest.TestCase):
    def _make_args(self, **overrides):
        args = types.SimpleNamespace(
            blocks_to_swap=0,
            lycoris=False,
            lora_weight=[],
            lora_multiplier=[],
            include_patterns=None,
            exclude_patterns=None,
            fp8_scaled=False,
            save_merged_model=None,
            disable_numpy_memmap=False,
            dit="/tmp/dit.safetensors",
            attn_mode="torch",
            use_32bit_attention=False,
            compile=False,
            use_pinned_memory_for_block_swap=False,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_parse_args_normalizes_lora_multipliers(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "zimage_generate_image.py",
                "--text_encoder",
                "/tmp/text_encoder",
                "--save_path",
                "/tmp/output",
                "--prompt",
                "hello",
                "--lora_weight",
                "a.safetensors",
                "b.safetensors",
                "--lora_multiplier",
                "0.5",
            ],
        ):
            args = zimage_generate_image.parse_args()

        self.assertEqual(args.lora_weight, ["a.safetensors", "b.safetensors"])
        self.assertEqual(args.lora_multiplier, [0.5, 1.0])

    def test_load_dit_model_normalizes_zimage_inference_lora_weights(self):
        args = self._make_args(lora_weight=["dummy.safetensors"], lora_multiplier=[1.0])
        raw_weights = {
            "lora_unet_transformer_blocks_0_attention_qkv.lora_down.weight": torch.ones(2, 4),
            "lora_unet_transformer_blocks_0_attention_qkv.lora_up.weight": torch.ones(12, 2),
        }
        captured = {}

        def fake_filter(weights_sd, *_args, **_kwargs):
            captured["keys"] = set(weights_sd.keys())
            return weights_sd

        with (
            mock.patch.object(zimage_generate_image, "load_file", return_value=raw_weights),
            mock.patch.object(zimage_generate_image, "filter_lora_state_dict", side_effect=fake_filter),
            mock.patch.object(zimage_generate_image.zimage_model, "load_zimage_model", return_value=_DummyZImageModel()),
            mock.patch.object(zimage_generate_image, "clean_memory_on_device"),
        ):
            zimage_generate_image.load_dit_model(args, torch.device("cpu"), dit_weight_dtype=None)

        self.assertIn("lora_unet_transformer_blocks_0_attention_to_q.lora_down.weight", captured["keys"])
        self.assertIn("lora_unet_transformer_blocks_0_attention_to_k.lora_up.weight", captured["keys"])
        self.assertIn("lora_unet_transformer_blocks_0_attention_to_v.alpha", captured["keys"])

    def test_load_dit_model_passes_converter_for_lycoris(self):
        args = self._make_args(lycoris=True, lora_weight=["dummy.safetensors"], lora_multiplier=[1.0])

        with (
            mock.patch.object(zimage_generate_image.zimage_model, "load_zimage_model", return_value=_DummyZImageModel()),
            mock.patch.object(zimage_generate_image, "merge_lora_weights") as merge_lora_weights,
            mock.patch.object(zimage_generate_image, "clean_memory_on_device"),
        ):
            zimage_generate_image.load_dit_model(args, torch.device("cpu"), dit_weight_dtype=None)

        self.assertEqual(merge_lora_weights.call_count, 1)
        self.assertIs(
            merge_lora_weights.call_args.kwargs["converter"],
            zimage_generate_image.normalize_zimage_inference_lora_weights,
        )


if __name__ == "__main__":
    unittest.main()
