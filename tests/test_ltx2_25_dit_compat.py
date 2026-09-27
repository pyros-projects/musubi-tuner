import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musubi_tuner.ltx_2.model.transformer.model import LTXModel, LTXModelType
from musubi_tuner.ltx_2.model.transformer.model_configurator import LTXModelConfigurator

TINY = dict(
    model_type=LTXModelType.AudioVideo,
    num_attention_heads=2,
    attention_head_dim=4,
    in_channels=8,
    out_channels=8,
    num_layers=2,
    cross_attention_dim=8,
    caption_channels=8,
    audio_num_attention_heads=2,
    audio_attention_head_dim=2,
    audio_in_channels=4,
    audio_out_channels=4,
    audio_cross_attention_dim=4,
    caption_proj_before_connector=True,
    cross_attention_adaln=True,
    apply_gated_attention=True,
)


def _tiny_model(**overrides) -> LTXModel:
    kwargs = dict(TINY)
    kwargs.update(overrides)
    return LTXModel(**kwargs)


class LTX25FFBiasTests(unittest.TestCase):
    def test_default_ff_bias_true_builds_video_and_audio_ff_biases(self):
        model = _tiny_model()
        keys = set(model.state_dict().keys())
        self.assertIn("transformer_blocks.0.ff.net.0.proj.bias", keys)
        self.assertIn("transformer_blocks.0.ff.net.2.bias", keys)
        self.assertIn("transformer_blocks.0.audio_ff.net.0.proj.bias", keys)
        self.assertIn("transformer_blocks.0.audio_ff.net.2.bias", keys)

    def test_ff_bias_false_drops_video_ff_biases_but_keeps_audio(self):
        model = _tiny_model(ff_bias=False)
        keys = set(model.state_dict().keys())
        for idx in range(2):
            self.assertNotIn(f"transformer_blocks.{idx}.ff.net.0.proj.bias", keys)
            self.assertNotIn(f"transformer_blocks.{idx}.ff.net.2.bias", keys)
            # LTX-2.5 checkpoints keep audio FF biases even with ff_bias=false
            self.assertIn(f"transformer_blocks.{idx}.audio_ff.net.0.proj.bias", keys)
            self.assertIn(f"transformer_blocks.{idx}.audio_ff.net.2.bias", keys)
        # unrelated biases untouched
        self.assertIn("patchify_proj.bias", keys)
        self.assertIn("transformer_blocks.0.ff.net.0.proj.weight", keys)


class LTX25KeyframesTests(unittest.TestCase):
    def test_keyframes_param_absent_by_default(self):
        model = _tiny_model()
        self.assertNotIn("keyframes_abs_pos_embedding", model.state_dict())

    def test_keyframes_param_created_with_inner_dim_shape(self):
        model = _tiny_model(use_keyframes_abs_pos_embedding=True)
        sd = model.state_dict()
        self.assertIn("keyframes_abs_pos_embedding", sd)
        self.assertEqual(tuple(sd["keyframes_abs_pos_embedding"].shape), (1, model.inner_dim))

    def test_25_flavored_state_dict_round_trips_strict(self):
        source = _tiny_model(ff_bias=False, use_keyframes_abs_pos_embedding=True)
        sd = {k: torch.randn_like(v) for k, v in source.state_dict().items()}
        target = _tiny_model(ff_bias=False, use_keyframes_abs_pos_embedding=True)
        target.load_state_dict(sd, strict=True)
        self.assertTrue(torch.equal(target.state_dict()["keyframes_abs_pos_embedding"], sd["keyframes_abs_pos_embedding"]))


class LTX25ConfiguratorTests(unittest.TestCase):
    CONFIG_25_TINY = {
        "transformer": {
            "dropout": 0.0,
            "attention_bias": True,
            "num_vector_embeds": None,
            "activation_fn": "gelu-approximate",
            "num_embeds_ada_norm": 1000,
            "use_linear_projection": False,
            "only_cross_attention": False,
            "cross_attention_norm": True,
            "double_self_attention": False,
            "upcast_attention": False,
            "standardization_norm": "rms_norm",
            "norm_elementwise_affine": False,
            "qk_norm": "rms_norm",
            "positional_embedding_type": "rope",
            "use_audio_video_cross_attention": True,
            "share_ff": False,
            "av_cross_ada_norm": True,
            "use_middle_indices_grid": True,
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "in_channels": 8,
            "out_channels": 8,
            "num_layers": 2,
            "cross_attention_dim": 8,
            "caption_channels": 8,
            "audio_num_attention_heads": 2,
            "audio_attention_head_dim": 2,
            "audio_in_channels": 4,
            "audio_out_channels": 4,
            "audio_cross_attention_dim": 4,
            "caption_proj_before_connector": True,
            "cross_attention_adaln": True,
            "apply_gated_attention": True,
            "rope_type": "split",
            "frequencies_precision": "float64",
            # LTX-2.5 additions
            "ff_bias": False,
            "use_keyframes_abs_pos_embedding": True,
            "text_encoder_norm_type": "PER_TOKEN_RMS",
        }
    }

    def test_from_config_reads_25_keys(self):
        model = LTXModelConfigurator.from_config(self.CONFIG_25_TINY)
        keys = set(model.state_dict().keys())
        self.assertNotIn("transformer_blocks.0.ff.net.0.proj.bias", keys)
        self.assertIn("transformer_blocks.0.audio_ff.net.0.proj.bias", keys)
        self.assertIn("keyframes_abs_pos_embedding", keys)
        self.assertFalse(model.ff_bias)
        self.assertTrue(model.use_keyframes_abs_pos_embedding)

    def test_from_config_23_defaults_unchanged(self):
        config = {"transformer": dict(self.CONFIG_25_TINY["transformer"])}
        for key in ("ff_bias", "use_keyframes_abs_pos_embedding", "text_encoder_norm_type"):
            config["transformer"].pop(key)
        model = LTXModelConfigurator.from_config(config)
        keys = set(model.state_dict().keys())
        self.assertIn("transformer_blocks.0.ff.net.0.proj.bias", keys)
        self.assertNotIn("keyframes_abs_pos_embedding", keys)
        self.assertTrue(model.ff_bias)


if __name__ == "__main__":
    unittest.main()
