import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from musubi_tuner.ltx2_repack import repack


def _make_transformer(path: Path) -> dict[str, torch.Tensor]:
    tensors = {
        "model.diffusion_model.patchify_proj.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        "model.diffusion_model.keyframes_abs_pos_embedding": torch.randn(1, 4, dtype=torch.bfloat16),
    }
    config = {
        "transformer": {"num_layers": 1, "ff_bias": False, "use_keyframes_abs_pos_embedding": True},
        "scheduler": {"_class_name": "RectifiedFlowScheduler"},
    }
    save_file(
        tensors,
        path,
        metadata={
            "config": json.dumps(config),
            "model_version": "2.5.0",
            "gemma_source_checkpoint": json.dumps({"gemma_version": "gemma4-12b-ltx-v1"}),
            "license": "LTX-2.x Community License",
        },
    )
    return tensors


def _make_video_vae(path: Path) -> dict[str, torch.Tensor]:
    tensors = {
        "encoder.conv_in.conv.weight": torch.randn(3, 3, dtype=torch.bfloat16),
        "decoder.conv_out.conv.weight": torch.randn(3, 3, dtype=torch.bfloat16),
        "per_channel_statistics.std-of-means": torch.randn(4, dtype=torch.float32),
    }
    config = {"vae": {"_class_name": "CausalVideoAutoencoder", "encoder_blocks": [["res_x", {"num_layers": 1}]]}}
    save_file(tensors, path, metadata={"config": json.dumps(config), "model_version": "2.5.0"})
    return tensors


def _make_audio_vae(path: Path) -> dict[str, torch.Tensor]:
    tensors = {
        "audio_vae.encoder.weight": torch.randn(2, 2, dtype=torch.bfloat16),
        "vocoder.head.weight": torch.randn(2, 2, dtype=torch.bfloat16),
    }
    config = {"audio_vae": {"model": {"sample_rate": 44100}, "preprocessing": {}}, "vocoder": {"vocoder": {}, "bwe": {}}}
    save_file(tensors, path, metadata={"config": json.dumps(config)})
    return tensors


def _make_text_encoder(path: Path) -> dict[str, torch.Tensor]:
    tensors = {
        "text_embedding_projection.video_aggregate_embed.weight": torch.randn(4, 6, dtype=torch.bfloat16),
        "text_embedding_projection.video_aggregate_embed.bias": torch.randn(4, dtype=torch.bfloat16),
        "text_embedding_projection.audio_aggregate_embed.weight": torch.randn(2, 6, dtype=torch.bfloat16),
        "text_embedding_projection.audio_aggregate_embed.bias": torch.randn(2, dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        "tokenizer_json": torch.zeros(16, dtype=torch.uint8),
    }
    save_file(tensors, path, metadata={"gemma_config": json.dumps({"model_type": "gemma4_unified"})})
    return tensors


class LTX2RepackTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.transformer = self.tmp / "transformer.safetensors"
        self.video_vae = self.tmp / "video_vae.safetensors"
        self.audio_vae = self.tmp / "audio_vae.safetensors"
        self.text_encoder = self.tmp / "text_encoder.safetensors"
        self.output = self.tmp / "single.safetensors"
        self.transformer_tensors = _make_transformer(self.transformer)
        self.video_vae_tensors = _make_video_vae(self.video_vae)
        self.audio_vae_tensors = _make_audio_vae(self.audio_vae)
        self.text_encoder_tensors = _make_text_encoder(self.text_encoder)

    def tearDown(self):
        self._tmp.cleanup()

    def _repack(self, **overrides):
        kwargs = dict(
            transformer=str(self.transformer),
            video_vae=str(self.video_vae),
            audio_vae=str(self.audio_vae),
            text_encoder=str(self.text_encoder),
            output=str(self.output),
        )
        kwargs.update(overrides)
        return repack(**kwargs)

    def test_repack_produces_23_layout_with_merged_config(self):
        counts = self._repack()
        self.assertEqual(counts, {"transformer": 3, "video_vae": 3, "audio_vae": 2, "text_projection": 4})

        expected = dict(self.transformer_tensors)
        expected.update({f"vae.{k}": v for k, v in self.video_vae_tensors.items()})
        expected.update(self.audio_vae_tensors)
        expected.update({k: v for k, v in self.text_encoder_tensors.items() if k.startswith("text_embedding_projection.")})

        with safe_open(self.output, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            keys = set(f.keys())
            self.assertEqual(keys, set(expected))
            for key, tensor in expected.items():
                loaded = f.get_tensor(key)
                self.assertEqual(loaded.dtype, tensor.dtype, key)
                self.assertTrue(torch.equal(loaded, tensor), key)

        config = json.loads(metadata["config"])
        self.assertEqual(sorted(config), ["audio_vae", "scheduler", "transformer", "vae", "vocoder"])
        self.assertEqual(config["transformer"]["ff_bias"], False)
        self.assertEqual(config["vae"]["encoder_blocks"], [["res_x", {"num_layers": 1}]])
        self.assertEqual(metadata["model_version"], "2.5.0")
        self.assertIn("gemma4-12b-ltx-v1", metadata["gemma_source_checkpoint"])
        self.assertIn("License", metadata["license"])

    def test_text_encoder_contributes_only_projections(self):
        self._repack()
        with safe_open(self.output, framework="pt", device="cpu") as f:
            keys = set(f.keys())
        self.assertNotIn("model.embed_tokens.weight", keys)
        self.assertNotIn("tokenizer_json", keys)
        self.assertEqual(sum(k.startswith("text_embedding_projection.") for k in keys), 4)

    def test_refuses_to_overwrite_without_flag(self):
        self._repack()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self._repack()
        self._repack(overwrite=True)  # succeeds

    def test_rejects_wrong_file_in_transformer_slot(self):
        with self.assertRaisesRegex(ValueError, "model.diffusion_model"):
            self._repack(transformer=str(self.video_vae))

    def test_rejects_wrong_file_in_video_vae_slot(self):
        with self.assertRaisesRegex(ValueError, "wrong file passed to --video_vae"):
            self._repack(video_vae=str(self.transformer))

    def test_rejects_diffusion_decoder_vae_config(self):
        path = self.tmp / "diffusion_vae.safetensors"
        tensors = {
            "encoder.conv_in.conv.weight": torch.randn(3, 3, dtype=torch.bfloat16),
            "decoder.diff_blocks.0.mlp.w_up.weight": torch.randn(3, 3, dtype=torch.bfloat16),
        }
        config = {"vae": {"_class_name": "CausalDiffusionVAE", "encoder": {"blocks": []}, "decoder": {"blocks": []}}}
        save_file(tensors, path, metadata={"config": json.dumps(config)})
        with self.assertRaisesRegex(ValueError, "-conv"):
            self._repack(video_vae=str(path))

    def test_rejects_quantized_transformer(self):
        # int8 dtype WITHOUT comfy_quant markers is malformed, not a valid int8-convrot file.
        path = self.tmp / "int8.safetensors"
        tensors = {
            "model.diffusion_model.block.weight": torch.zeros(2, 2, dtype=torch.int8),
        }
        config = {"transformer": {"num_layers": 1}}
        save_file(tensors, path, metadata={"config": json.dumps(config)})
        with self.assertRaisesRegex(ValueError, "quantized|bf16"):
            self._repack(transformer=str(path))

    @staticmethod
    def _marker_tensor(marker: dict) -> torch.Tensor:
        return torch.frombuffer(bytearray(json.dumps(marker).encode("utf-8")), dtype=torch.uint8).clone()

    def _make_int8_transformer(self, path: Path, marker: dict) -> dict[str, torch.Tensor]:
        tensors = {
            "model.diffusion_model.patchify_proj.weight": torch.randn(4, 4, dtype=torch.bfloat16),
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": torch.randint(
                -128, 127, (8, 4), dtype=torch.int8
            ),
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale": torch.rand(8, 1, dtype=torch.float32),
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.comfy_quant": self._marker_tensor(marker),
            "model.diffusion_model.keyframes_abs_pos_embedding": torch.randn(1, 4, dtype=torch.bfloat16),
        }
        config = {
            "transformer": {"num_layers": 1, "ff_bias": False, "use_keyframes_abs_pos_embedding": True},
            "scheduler": {"_class_name": "RectifiedFlowScheduler"},
        }
        save_file(
            tensors,
            path,
            metadata={"config": json.dumps(config), "model_version": "2.5.0", "format": "pt"},
        )
        return tensors

    def test_accepts_int8_convrot_transformer_verbatim(self):
        path = self.tmp / "transformer_int8.safetensors"
        tensors = self._make_int8_transformer(
            path, {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
        )
        counts = self._repack(transformer=str(path))
        self.assertEqual(counts["transformer"], 5)

        with safe_open(self.output, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for key, tensor in tensors.items():
                loaded = f.get_tensor(key)
                self.assertEqual(loaded.dtype, tensor.dtype, key)
                self.assertTrue(torch.equal(loaded, tensor), key)
        self.assertEqual(metadata["format"], "pt")

    def test_rejects_nvfp4_marker_transformer(self):
        path = self.tmp / "transformer_nvfp4.safetensors"
        self._make_int8_transformer(path, {"format": "nvfp4", "convrot": False})
        with self.assertRaisesRegex(ValueError, "int8-convrot"):
            self._repack(transformer=str(path))

    def test_rejects_quantized_video_vae(self):
        path = self.tmp / "video_vae_int8.safetensors"
        tensors = {
            "encoder.conv_in.conv.weight": torch.randn(3, 3, dtype=torch.bfloat16),
            "decoder.conv_out.conv.weight": torch.zeros(3, 3, dtype=torch.int8),
            "decoder.conv_out.conv.weight_scale": torch.rand(3, 1, dtype=torch.float32),
        }
        config = {"vae": {"_class_name": "CausalVideoAutoencoder", "encoder_blocks": [["res_x", {"num_layers": 1}]]}}
        save_file(tensors, path, metadata={"config": json.dumps(config)})
        with self.assertRaisesRegex(ValueError, "bf16"):
            self._repack(video_vae=str(path))


if __name__ == "__main__":
    unittest.main()
