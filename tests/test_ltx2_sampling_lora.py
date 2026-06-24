import argparse
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musubi_tuner.ltx2_generate_video import (
    _apply_official_distilled_pipeline_args,
    _build_prompt_list,
    _merge_lora_weights,
    _should_merge_cli_loras_once,
    main,
    parse_args,
)
from musubi_tuner.ltx2_inference import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    InferenceConfig,
    LTX2Inferencer,
    build_vae_tiling_config,
)
from musubi_tuner.ltx2_lora_utils import normalize_ltx_comfy_lora_weights
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer, ltx2_setup_parser
from musubi_tuner.hv_train_network import resolve_step_sampling_requests, setup_parser_common
from musubi_tuner.ltx_2.model.video_vae.video_vae import VideoDecoder, resolve_slice_bounds
from musubi_tuner.ltx_2.model.video_vae.tiling import Tile


class _FakeAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return nullcontext()


class _FakePartialState:
    def __init__(self):
        self.num_processes = 1


class _FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.blocks_to_swap = 0
        self.inference_switches = 0
        self.training_switches = 0

    def switch_block_swap_for_inference(self):
        self.inference_switches += 1

    def switch_block_swap_for_training(self):
        self.training_switches += 1


class _FakeCompileTransformer(_FakeTransformer):
    def __init__(self):
        super().__init__()
        self._orig_mod = self


class _RecordingTrainer(LTX2NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.applied = []
        self.restored = []
        self.sampled = []

    def _apply_sampling_lora(self, args, transformer, device):
        self.applied.append((tuple(args.sampling_lora_weight), device.type))
        return [("dummy.safetensors", {})]

    def _restore_sampling_lora(self, args, transformer, device, applied):
        self.restored.append((tuple(args.sampling_lora_weight), len(applied), device.type))

    def sample_image_inference(
        self,
        accelerator,
        args,
        transformer,
        dit_dtype,
        vae,
        save_dir,
        sample_parameter,
        epoch,
        steps,
        audio_decoder=None,
        vocoder=None,
    ):
        self.sampled.append(sample_parameter["prompt"])


class _SamplingHooksTrainer(LTX2NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.events = []

    def _get_sampling_lora_specs(self, args):
        return [("dummy.safetensors", 0.8)]

    def _sampling_lora_requires_eager_fallback(self, transformer):
        return False

    def _apply_sampling_lora(self, args, transformer, device):
        self.events.append(("apply", device.type))
        return [("dummy.safetensors", {})]

    def _restore_sampling_lora(self, args, transformer, device, applied):
        self.events.append(("restore", len(applied), device.type))

    def sample_image_inference(
        self,
        accelerator,
        args,
        transformer,
        dit_dtype,
        vae,
        save_dir,
        sample_parameter,
        epoch,
        steps,
        audio_decoder=None,
        vocoder=None,
    ):
        self.events.append(("sample", sample_parameter["prompt"]))


class _PromptLocalSamplingTrainer(LTX2NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.events = []

    def _apply_sampling_lora_specs(self, args, transformer, device, specs):
        if not specs:
            return []
        self.events.append(("apply_specs", tuple(specs), device.type))
        return [("prompt-local", {"backups": len(specs)})]

    def _restore_sampling_lora_specs(self, args, transformer, device, applied):
        if not applied:
            return
        self.events.append(("restore_specs", tuple(applied), device.type))

    def sample_image_inference(
        self,
        accelerator,
        args,
        transformer,
        dit_dtype,
        vae,
        save_dir,
        sample_parameter,
        epoch,
        steps,
        audio_decoder=None,
        vocoder=None,
    ):
        self.events.append(("sample", sample_parameter["prompt"]))


class _AutocacheSamplingTrainer(LTX2NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.events = []

    def sample_image_inference(
        self,
        accelerator,
        args,
        transformer,
        dit_dtype,
        vae,
        save_dir,
        sample_parameter,
        epoch,
        steps,
        audio_decoder=None,
        vocoder=None,
    ):
        self.events.append(("sample", sample_parameter["prompt"]))


class _FakeSamplingLora:
    def __init__(self, org_module: torch.nn.Module):
        self.lora_name = "lora_unet_model_linear"
        self.org_module_ref = [org_module]
        self.multiplier = 1.0
        self.scale = 1.0
        self.split_dims = None


class _Float8OverlayTrainer(LTX2NetworkTrainer):
    def _sampling_lora_float8_dtypes(self):
        return (torch.float8_e4m3fn,)


class _FakeCaptionProjection:
    def __init__(self, in_features: int):
        self.linear_1 = type("Linear1", (), {"in_features": in_features})()


class _FakeUpsampler:
    def to(self, device):
        return self


class LTX2SamplingLoraTests(unittest.TestCase):
    def test_two_stage_generate_applies_distilled_lora_only_at_stage2_by_default(self):
        inferencer = LTX2Inferencer(
            transformer=_FakeTransformer(),
            vae=object(),
            device=torch.device("cpu"),
            dit_dtype=torch.float32,
            audio_video_mode=False,
        )
        inferencer._spatial_upsampler = _FakeUpsampler()
        inferencer._distilled_lora_state = {"dummy": torch.randn(1)}

        events = []

        def _record_denoise(latents, sigmas, *args, **kwargs):
            events.append(f"denoise:{kwargs.get('progress_desc')}")
            return latents, None

        with mock.patch.object(inferencer, "_prepare_prompt_embeds", return_value=(torch.zeros(1, 1, 4), None)), \
             mock.patch.object(inferencer, "_get_vae_factors", return_value=(8, 32)), \
             mock.patch.object(inferencer, "_get_expected_embed_dim", return_value=None), \
             mock.patch.object(inferencer, "_init_latents", return_value=torch.zeros((1, 1, 7, 2, 2))), \
             mock.patch.object(inferencer, "_upsample_latents", side_effect=lambda latents, upsampler: (events.append("upsample"), latents)[1]), \
             mock.patch.object(inferencer, "_apply_distilled_lora", side_effect=lambda multiplier=1.0: events.append("apply")), \
             mock.patch.object(inferencer, "_remove_distilled_lora", side_effect=lambda multiplier=1.0: events.append("remove")), \
             mock.patch.object(inferencer, "_denoise_loop", side_effect=_record_denoise):
            inferencer.generate(
                InferenceConfig(
                    two_stage=True,
                    spatial_upsampler_path="/tmp/upsampler.safetensors",
                    distilled_lora_path="/tmp/distilled.safetensors",
                    width=768,
                    height=512,
                    frame_count=49,
                    sample_steps=2,
                    stage2_steps=1,
                    cfg_scale=1.0,
                    guidance_scale=1.0,
                    prompt_embeds=torch.zeros(1, 1, 4),
                ),
                decode_video=False,
            )

        self.assertEqual(
            events,
            ["denoise:Stage 1", "upsample", "apply", "denoise:Stage 2 refine", "remove"],
        )

    def test_two_stage_generate_can_apply_distilled_lora_from_stage1(self):
        inferencer = LTX2Inferencer(
            transformer=_FakeTransformer(),
            vae=object(),
            device=torch.device("cpu"),
            dit_dtype=torch.float32,
            audio_video_mode=False,
        )
        inferencer._spatial_upsampler = _FakeUpsampler()
        inferencer._distilled_lora_state = {"dummy": torch.randn(1)}

        events = []

        def _record_denoise(latents, sigmas, *args, **kwargs):
            events.append(f"denoise:{kwargs.get('progress_desc')}")
            return latents, None

        with mock.patch.object(inferencer, "_prepare_prompt_embeds", return_value=(torch.zeros(1, 1, 4), None)), \
             mock.patch.object(inferencer, "_get_vae_factors", return_value=(8, 32)), \
             mock.patch.object(inferencer, "_get_expected_embed_dim", return_value=None), \
             mock.patch.object(inferencer, "_init_latents", return_value=torch.zeros((1, 1, 7, 2, 2))), \
             mock.patch.object(inferencer, "_upsample_latents", side_effect=lambda latents, upsampler: (events.append("upsample"), latents)[1]), \
             mock.patch.object(inferencer, "_apply_distilled_lora", side_effect=lambda multiplier=1.0: events.append("apply")), \
             mock.patch.object(inferencer, "_remove_distilled_lora", side_effect=lambda multiplier=1.0: events.append("remove")), \
             mock.patch.object(inferencer, "_denoise_loop", side_effect=_record_denoise):
            inferencer.generate(
                InferenceConfig(
                    two_stage=True,
                    spatial_upsampler_path="/tmp/upsampler.safetensors",
                    distilled_lora_path="/tmp/distilled.safetensors",
                    stage1_use_distilled_lora=True,
                    width=768,
                    height=512,
                    frame_count=49,
                    sample_steps=2,
                    stage2_steps=1,
                    cfg_scale=1.0,
                    guidance_scale=1.0,
                    prompt_embeds=torch.zeros(1, 1, 4),
                ),
                decode_video=False,
            )

        self.assertEqual(
            events,
            ["apply", "denoise:Stage 1", "upsample", "denoise:Stage 2 refine", "remove"],
        )

    def test_two_stage_generate_uses_distilled_lora_stage_specific_multipliers(self):
        inferencer = LTX2Inferencer(
            transformer=_FakeTransformer(),
            vae=object(),
            device=torch.device("cpu"),
            dit_dtype=torch.float32,
            audio_video_mode=False,
        )
        inferencer._spatial_upsampler = _FakeUpsampler()
        inferencer._distilled_lora_state = {"dummy": torch.randn(1)}

        events = []

        def _record_apply(multiplier=1.0):
            events.append(("apply", float(multiplier)))

        def _record_remove(multiplier=1.0):
            events.append(("remove", float(multiplier)))

        with mock.patch.object(inferencer, "_prepare_prompt_embeds", return_value=(torch.zeros(1, 1, 4), None)), \
             mock.patch.object(inferencer, "_get_vae_factors", return_value=(8, 32)), \
             mock.patch.object(inferencer, "_get_expected_embed_dim", return_value=None), \
             mock.patch.object(inferencer, "_init_latents", return_value=torch.zeros((1, 1, 7, 2, 2))), \
             mock.patch.object(inferencer, "_upsample_latents", side_effect=lambda latents, upsampler: latents), \
             mock.patch.object(inferencer, "_apply_distilled_lora", side_effect=_record_apply), \
             mock.patch.object(inferencer, "_remove_distilled_lora", side_effect=_record_remove), \
             mock.patch.object(inferencer, "_denoise_loop", side_effect=lambda latents, sigmas, *args, **kwargs: (latents, None)):
            inferencer.generate(
                InferenceConfig(
                    two_stage=True,
                    spatial_upsampler_path="/tmp/upsampler.safetensors",
                    distilled_lora_path="/tmp/distilled.safetensors",
                    stage1_use_distilled_lora=True,
                    stage1_distilled_lora_multiplier=0.65,
                    stage2_distilled_lora_multiplier=0.35,
                    width=768,
                    height=512,
                    frame_count=49,
                    sample_steps=2,
                    stage2_steps=1,
                    cfg_scale=1.0,
                    guidance_scale=1.0,
                    prompt_embeds=torch.zeros(1, 1, 4),
                ),
                decode_video=False,
            )

        self.assertEqual(
            events,
            [("apply", 0.65), ("remove", 0.65), ("apply", 0.35), ("remove", 0.35)],
        )

        events.clear()
        with mock.patch.object(inferencer, "_prepare_prompt_embeds", return_value=(torch.zeros(1, 1, 4), None)), \
             mock.patch.object(inferencer, "_get_vae_factors", return_value=(8, 32)), \
             mock.patch.object(inferencer, "_get_expected_embed_dim", return_value=None), \
             mock.patch.object(inferencer, "_init_latents", return_value=torch.zeros((1, 1, 7, 2, 2))), \
             mock.patch.object(inferencer, "_upsample_latents", side_effect=lambda latents, upsampler: latents), \
             mock.patch.object(inferencer, "_apply_distilled_lora", side_effect=_record_apply), \
             mock.patch.object(inferencer, "_remove_distilled_lora", side_effect=_record_remove), \
             mock.patch.object(inferencer, "_denoise_loop", side_effect=lambda latents, sigmas, *args, **kwargs: (latents, None)):
            inferencer.generate(
                InferenceConfig(
                    two_stage=True,
                    spatial_upsampler_path="/tmp/upsampler.safetensors",
                    distilled_lora_path="/tmp/distilled.safetensors",
                    stage1_use_distilled_lora=False,
                    stage1_distilled_lora_multiplier=0.65,
                    stage2_distilled_lora_multiplier=0.35,
                    width=768,
                    height=512,
                    frame_count=49,
                    sample_steps=2,
                    stage2_steps=1,
                    cfg_scale=1.0,
                    guidance_scale=1.0,
                    prompt_embeds=torch.zeros(1, 1, 4),
                ),
                decode_video=False,
            )

        self.assertEqual(events, [("apply", 0.35), ("remove", 0.35)])

    def test_two_stage_generate_uses_explicit_stage_sigmas(self):
        inferencer = LTX2Inferencer(
            transformer=_FakeTransformer(),
            vae=object(),
            device=torch.device("cpu"),
            dit_dtype=torch.float32,
            audio_video_mode=False,
        )
        inferencer._spatial_upsampler = _FakeUpsampler()

        recorded_sigmas = []

        def _record_denoise(latents, sigmas, *args, **kwargs):
            recorded_sigmas.append(sigmas.detach().cpu().tolist())
            return latents, None

        with mock.patch.object(inferencer, "_prepare_prompt_embeds", return_value=(torch.zeros(1, 1, 4), None)), \
             mock.patch.object(inferencer, "_get_vae_factors", return_value=(8, 32)), \
             mock.patch.object(inferencer, "_get_expected_embed_dim", return_value=None), \
             mock.patch.object(inferencer, "_init_latents", return_value=torch.zeros((1, 1, 7, 2, 2))), \
             mock.patch.object(inferencer, "_upsample_latents", side_effect=lambda latents, upsampler: latents), \
             mock.patch.object(inferencer, "_denoise_loop", side_effect=_record_denoise):
            inferencer.generate(
                InferenceConfig(
                    two_stage=True,
                    spatial_upsampler_path="/tmp/upsampler.safetensors",
                    width=768,
                    height=512,
                    frame_count=49,
                    sample_steps=2,
                    cfg_scale=1.0,
                    guidance_scale=1.0,
                    prompt_embeds=torch.zeros(1, 1, 4),
                    stage1_sigmas=[1.0, 0.5, 0.0],
                    stage2_sigmas=[0.9, 0.4, 0.0],
                ),
                decode_video=False,
            )

        self.assertEqual(recorded_sigmas[0], [1.0, 0.5, 0.0])
        for got, expected in zip(recorded_sigmas[1], [0.9, 0.4, 0.0]):
            self.assertAlmostEqual(got, expected, places=6)

    def test_two_stage_generate_offloads_transformer_before_vae_decode_when_requested(self):
        class _RecordingTransformer(_FakeTransformer):
            def __init__(self, events):
                super().__init__()
                self.events = events
                self.move_calls = 0

            def move_to_device_except_swap_blocks(self, device):
                self.move_calls += 1
                phase = {
                    1: "transformer:offload_for_upsample",
                    2: "transformer:restore_for_stage2",
                    3: "transformer:offload_for_decode",
                }.get(self.move_calls, f"transformer:move_{self.move_calls}")
                self.events.append(phase)
                return self

        class _RecordingVAE:
            def __init__(self, events):
                self.events = events
                self.device = torch.device("cpu")
                self.dtype = torch.float32

            def to_device(self, device):
                self.device = torch.device(device)
                self.events.append(f"vae:{self.device}")

            def to_dtype(self, dtype):
                self.dtype = dtype

            def decode(self, items):
                self.events.append("decode")
                return [torch.zeros((3, 7, 8, 8), dtype=torch.float32)]

        events = []
        inferencer = LTX2Inferencer(
            transformer=_RecordingTransformer(events),
            vae=_RecordingVAE(events),
            device=torch.device("cpu"),
            dit_dtype=torch.float32,
            audio_video_mode=False,
        )
        inferencer._spatial_upsampler = _FakeUpsampler()

        def _record_denoise(latents, sigmas, *args, **kwargs):
            events.append(f"denoise:{kwargs.get('progress_desc')}")
            return latents, None

        with mock.patch.object(inferencer, "_prepare_prompt_embeds", return_value=(torch.zeros(1, 1, 4), None)), \
             mock.patch.object(inferencer, "_get_vae_factors", return_value=(8, 32)), \
             mock.patch.object(inferencer, "_get_expected_embed_dim", return_value=None), \
             mock.patch.object(inferencer, "_init_latents", return_value=torch.zeros((1, 1, 7, 2, 2))), \
             mock.patch.object(inferencer, "_upsample_latents", side_effect=lambda latents, upsampler: latents), \
             mock.patch.object(inferencer, "_denoise_loop", side_effect=_record_denoise):
            inferencer.generate(
                InferenceConfig(
                    two_stage=True,
                    offload_between_stages=True,
                    spatial_upsampler_path="/tmp/upsampler.safetensors",
                    width=768,
                    height=512,
                    frame_count=49,
                    sample_steps=2,
                    stage2_steps=1,
                    cfg_scale=1.0,
                    guidance_scale=1.0,
                    prompt_embeds=torch.zeros(1, 1, 4),
                ),
                decode_video=True,
            )

        self.assertEqual(
            events[0:4],
            ["denoise:Stage 1", "transformer:offload_for_upsample", "transformer:restore_for_stage2", "denoise:Stage 2 refine"],
        )
        decode_idx = events.index("decode")
        self.assertLess(events.index("denoise:Stage 2 refine"), decode_idx)
        self.assertLess(events.index("transformer:offload_for_decode"), decode_idx)
        self.assertEqual(events[decode_idx - 1], "vae:cpu")

    def test_two_stage_generate_cleans_stage2_allocations_before_decode(self):
        class _RecordingVAE:
            def __init__(self, events):
                self.events = events

            def to_device(self, device):
                self.events.append(f"vae:{device}")

            def decode(self, items):
                self.events.append("decode")
                return [torch.zeros((3, 7, 8, 8), dtype=torch.float32)]

        events = []
        inferencer = LTX2Inferencer(
            transformer=_FakeTransformer(),
            vae=_RecordingVAE(events),
            device=torch.device("cpu"),
            dit_dtype=torch.float32,
            audio_video_mode=False,
        )
        inferencer._spatial_upsampler = _FakeUpsampler()

        def _record_denoise(latents, sigmas, *args, **kwargs):
            events.append(f"denoise:{kwargs.get('progress_desc')}")
            return latents, None

        with mock.patch.object(inferencer, "_prepare_prompt_embeds", return_value=(torch.zeros(1, 1, 4), None)), \
             mock.patch.object(inferencer, "_get_vae_factors", return_value=(8, 32)), \
             mock.patch.object(inferencer, "_get_expected_embed_dim", return_value=None), \
             mock.patch.object(inferencer, "_init_latents", return_value=torch.zeros((1, 1, 7, 2, 2))), \
             mock.patch.object(inferencer, "_upsample_latents", side_effect=lambda latents, upsampler: latents), \
             mock.patch.object(inferencer, "_denoise_loop", side_effect=_record_denoise), \
             mock.patch("musubi_tuner.ltx2_inference.clean_memory_on_device", side_effect=lambda device: events.append("cleanup")):
            inferencer.generate(
                InferenceConfig(
                    two_stage=True,
                    spatial_upsampler_path="/tmp/upsampler.safetensors",
                    width=768,
                    height=512,
                    frame_count=49,
                    sample_steps=2,
                    stage2_steps=1,
                    cfg_scale=1.0,
                    guidance_scale=1.0,
                    prompt_embeds=torch.zeros(1, 1, 4),
                ),
                decode_video=True,
            )

        decode_idx = events.index("decode")
        last_cleanup_before_decode = max(i for i, event in enumerate(events[:decode_idx]) if event == "cleanup")
        self.assertLess(events.index("denoise:Stage 2 refine"), last_cleanup_before_decode)

    def test_two_stage_generate_releases_stage2_noise_before_refine(self):
        events = []
        inferencer = LTX2Inferencer(
            transformer=_FakeTransformer(),
            vae=object(),
            device=torch.device("cpu"),
            dit_dtype=torch.float32,
            audio_video_mode=False,
        )
        inferencer._spatial_upsampler = _FakeUpsampler()

        def _record_denoise(latents, sigmas, *args, **kwargs):
            events.append(f"denoise:{kwargs.get('progress_desc')}")
            return latents, None

        with mock.patch.object(inferencer, "_prepare_prompt_embeds", return_value=(torch.zeros(1, 1, 4), None)), \
             mock.patch.object(inferencer, "_get_vae_factors", return_value=(8, 32)), \
             mock.patch.object(inferencer, "_get_expected_embed_dim", return_value=None), \
             mock.patch.object(inferencer, "_init_latents", return_value=torch.zeros((1, 1, 7, 2, 2))), \
             mock.patch.object(inferencer, "_upsample_latents", side_effect=lambda latents, upsampler: latents), \
             mock.patch.object(inferencer, "_denoise_loop", side_effect=_record_denoise), \
             mock.patch("musubi_tuner.ltx2_inference.clean_memory_on_device", side_effect=lambda device: events.append("cleanup")):
            inferencer.generate(
                InferenceConfig(
                    two_stage=True,
                    spatial_upsampler_path="/tmp/upsampler.safetensors",
                    width=768,
                    height=512,
                    frame_count=49,
                    sample_steps=2,
                    stage2_steps=1,
                    cfg_scale=1.0,
                    guidance_scale=1.0,
                    prompt_embeds=torch.zeros(1, 1, 4),
                ),
                decode_video=False,
            )

        refine_idx = events.index("denoise:Stage 2 refine")
        cleanup_before_refine = [i for i, event in enumerate(events[:refine_idx]) if event == "cleanup"]
        self.assertEqual(len(cleanup_before_refine), 2)

    def test_two_stage_generate_converts_av_cached_prompts_for_video_only_sampling(self):
        class _VideoOnlyTransformer(_FakeTransformer):
            def __init__(self):
                super().__init__()
                self.cross_attention_dim = 4096
                self.audio_cross_attention_dim = 2048

        inferencer = LTX2Inferencer(
            transformer=_VideoOnlyTransformer(),
            vae=object(),
            device=torch.device("cpu"),
            dit_dtype=torch.float32,
            audio_video_mode=False,
        )
        inferencer._spatial_upsampler = _FakeUpsampler()

        recorded_prompt_shapes = []

        def _record_denoise(latents, sigmas, prompt_embeds, prompt_mask, *args, **kwargs):
            recorded_prompt_shapes.append(tuple(prompt_embeds.shape))
            return latents, None

        av_prompt = torch.zeros((1, 3, 6144), dtype=torch.float32)
        av_negative = torch.zeros((1, 3, 6144), dtype=torch.float32)
        prompt_mask = torch.ones((1, 3), dtype=torch.int64)

        with mock.patch.object(inferencer, "_get_vae_factors", return_value=(8, 32)), \
             mock.patch.object(inferencer, "_init_latents", return_value=torch.zeros((1, 1, 7, 2, 2))), \
             mock.patch.object(inferencer, "_upsample_latents", side_effect=lambda latents, upsampler: latents), \
             mock.patch.object(inferencer, "_denoise_loop", side_effect=_record_denoise):
            inferencer.generate(
                InferenceConfig(
                    two_stage=True,
                    spatial_upsampler_path="/tmp/upsampler.safetensors",
                    width=768,
                    height=512,
                    frame_count=49,
                    sample_steps=2,
                    stage2_steps=1,
                    cfg_scale=3.0,
                    guidance_scale=3.0,
                    prompt_embeds=av_prompt,
                    prompt_attention_mask=prompt_mask,
                    negative_prompt_embeds=av_negative,
                    negative_prompt_attention_mask=prompt_mask,
                ),
                decode_video=False,
            )

        self.assertEqual(recorded_prompt_shapes, [(2, 3, 4096), (1, 3, 4096)])

    def test_distilled_lora_runtime_restore_preserves_existing_sampling_overlay(self):
        trainer = LTX2NetworkTrainer()
        transformer = _FakeTransformer()
        module = transformer.linear
        module._nf4_quantized = True

        trainer._attach_sampling_lora_runtime_overlay(
            module,
            torch.randn(2, 4),
            torch.randn(4, 2),
            0.5,
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
        )
        baseline_adapters = len(module._sampling_lora_runtime_adapters)

        inferencer = LTX2Inferencer(
            transformer=transformer,
            vae=object(),
            device=torch.device("cpu"),
            dit_dtype=torch.float32,
            audio_video_mode=False,
            sampling_lora_runtime_helper=trainer,
        )
        inferencer._distilled_lora_state = {
            "lora_unet_model_linear.lora_down.weight": torch.randn(2, 4),
            "lora_unet_model_linear.lora_up.weight": torch.randn(4, 2),
        }
        fake_network = type("Net", (), {"text_encoder_loras": [], "unet_loras": [_FakeSamplingLora(module)]})()

        with mock.patch(
            "musubi_tuner.networks.lora_ltx2.create_arch_network_from_weights",
            return_value=fake_network,
        ):
            inferencer._apply_distilled_lora()
            self.assertEqual(len(module._sampling_lora_runtime_adapters), baseline_adapters + 1)

            inferencer._remove_distilled_lora()

        self.assertEqual(len(module._sampling_lora_runtime_adapters), baseline_adapters)
        self.assertTrue(hasattr(module, "_sampling_lora_runtime_forward"))

    def test_inferencer_expected_embed_dim_uses_transformer_dims(self):
        transformer = type(
            "FakeTransformer",
            (),
            {
                "cross_attention_dim": 4096,
                "audio_cross_attention_dim": 2048,
            },
        )()
        inferencer = LTX2Inferencer(
            transformer=transformer,
            vae=object(),
            device=torch.device("cpu"),
            dit_dtype=torch.float32,
            audio_video_mode=False,
        )

        self.assertEqual(inferencer._get_expected_embed_dim(), 4096)

    def test_normalize_ltx_comfy_lora_weights_converts_keys(self):
        normalized = normalize_ltx_comfy_lora_weights(
            {
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight": torch.randn(2, 4),
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight": torch.randn(4, 2),
            }
        )

        self.assertIn(
            "lora_unet_model_transformer_blocks_0_attn1_to_k.lora_down.weight",
            normalized,
        )
        self.assertIn(
            "lora_unet_model_transformer_blocks_0_attn1_to_k.lora_up.weight",
            normalized,
        )
        self.assertIn(
            "lora_unet_model_transformer_blocks_0_attn1_to_k.alpha",
            normalized,
        )

    def test_resolve_slice_bounds_handles_open_ended_slice(self):
        self.assertEqual(resolve_slice_bounds(slice(0, None), 49), (0, 49))

    def test_accumulate_temporal_group_handles_open_ended_tile_slice(self):
        decoded_tile = torch.ones((1, 1, 49, 2, 2), dtype=torch.float32)
        fake_decoder = type(
            "FakeDecoder",
            (),
            {"forward": lambda self, latent_slice, timestep, generator: decoded_tile},
        )()
        tile = Tile(
            in_coords=(slice(None), slice(None), slice(None), slice(None), slice(None)),
            out_coords=(slice(None), slice(None), slice(0, None), slice(0, 2), slice(0, 2)),
            masks_1d=(None, None, None, None, None),
        )
        buffer = torch.zeros((1, 1, 49, 2, 2), dtype=torch.float32)
        latent = torch.zeros((1, 1, 7, 1, 1), dtype=torch.float32)

        weights = VideoDecoder._accumulate_temporal_group_into_buffer(
            fake_decoder,
            group_tiles=[tile],
            buffer=buffer,
            latent=latent,
            timestep=None,
            generator=None,
        )

        self.assertEqual(tuple(weights.shape), (1, 1, 49, 2, 2))
        self.assertTrue(torch.all(weights > 0))

    def test_build_vae_tiling_config_skips_temporal_config_when_disabled(self):
        tiling_config = build_vae_tiling_config(
            tile_size=512,
            tile_overlap=64,
            temporal_tile_size=0,
            temporal_tile_overlap=8,
        )

        self.assertIsNotNone(tiling_config)
        self.assertIsNotNone(tiling_config.spatial_config)
        self.assertIsNone(tiling_config.temporal_config)

    def test_generator_parse_args_accepts_ltx_version_flags(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "ltx2_generate_video.py",
                "--ltx2_checkpoint",
                "/tmp/ltx.safetensors",
                "--gemma_root",
                "/tmp/gemma",
                "--prompt",
                "hello",
                "--ltx_version",
                "2.3",
                "--ltx_version_check_mode",
                "error",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.ltx_version, "2.3")
        self.assertEqual(args.ltx_version_check_mode, "error")

    def test_generator_parse_args_accepts_stage1_distilled_flag(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "ltx2_generate_video.py",
                "--ltx2_checkpoint",
                "/tmp/ltx.safetensors",
                "--gemma_root",
                "/tmp/gemma",
                "--prompt",
                "hello",
                "--sample_stage1_use_distilled_lora",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.sample_stage1_use_distilled_lora)

    def test_generator_parse_args_accepts_official_distilled_pipeline_flag(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "ltx2_generate_video.py",
                "--ltx2_checkpoint",
                "/tmp/ltx.safetensors",
                "--gemma_root",
                "/tmp/gemma",
                "--prompt",
                "hello",
                "--sample_official_distilled_pipeline",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.sample_official_distilled_pipeline)

    def test_generator_parse_args_accepts_stage_specific_distilled_multipliers(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "ltx2_generate_video.py",
                "--ltx2_checkpoint",
                "/tmp/ltx.safetensors",
                "--gemma_root",
                "/tmp/gemma",
                "--prompt",
                "hello",
                "--sample_stage1_distilled_lora_multiplier",
                "0.75",
                "--sample_stage2_distilled_lora_multiplier",
                "0.25",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.sample_stage1_distilled_lora_multiplier, 0.75)
        self.assertEqual(args.sample_stage2_distilled_lora_multiplier, 0.25)

    def test_generator_parse_args_accepts_autocache_flag_without_value(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "ltx2_generate_video.py",
                "--ltx2_checkpoint",
                "/tmp/ltx.safetensors",
                "--gemma_root",
                "/tmp/gemma",
                "--prompt",
                "hello",
                "--autocache",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.autocache, "__cwd__")

    def test_generator_parse_args_accepts_autocache_explicit_path(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "ltx2_generate_video.py",
                "--ltx2_checkpoint",
                "/tmp/ltx.safetensors",
                "--gemma_root",
                "/tmp/gemma",
                "--prompt",
                "hello",
                "--autocache",
                "/tmp/cache",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.autocache, "/tmp/cache")

    def test_generator_parse_args_accepts_cache_prompt_file_and_defaults_autocache(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "ltx2_generate_video.py",
                "--ltx2_checkpoint",
                "/tmp/ltx.safetensors",
                "--gemma_root",
                "/tmp/gemma",
                "--prompt",
                "hello",
                "--cache_prompt_file",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.cache_prompt_file)
        self.assertEqual(args.autocache, "__cwd__")

    def test_training_parser_accepts_stage_specific_distilled_multipliers(self):
        parser = ltx2_setup_parser(argparse.ArgumentParser())

        args = parser.parse_args(
            [
                "--ltx2_checkpoint",
                "/tmp/ltx.safetensors",
                "--sample_stage1_distilled_lora_multiplier",
                "0.75",
                "--sample_stage2_distilled_lora_multiplier",
                "0.25",
            ]
        )

        self.assertEqual(args.sample_stage1_distilled_lora_multiplier, 0.75)
        self.assertEqual(args.sample_stage2_distilled_lora_multiplier, 0.25)

    def test_training_main_parser_accepts_sample_with_offloading_once(self):
        parser = ltx2_setup_parser(setup_parser_common())

        args = parser.parse_args(
            [
                "--ltx2_checkpoint",
                "/tmp/ltx.safetensors",
                "--sample_with_offloading",
            ]
        )

        self.assertTrue(args.sample_with_offloading)

    def test_apply_official_distilled_pipeline_args_sets_expected_overrides(self):
        args = Namespace(
            sample_official_distilled_pipeline=True,
            sample_two_stage=False,
            sample_stage1_use_distilled_lora=False,
            sample_stage2_steps=99,
            sample_sigmas=None,
            sample_steps=20,
            guidance_scale=3.5,
            cfg_scale=4.0,
            spatial_upsampler_path="/tmp/upscaler.safetensors",
            distilled_lora_path="/tmp/distilled.safetensors",
        )

        _apply_official_distilled_pipeline_args(args)

        self.assertTrue(args.sample_two_stage)
        self.assertTrue(args.sample_stage1_use_distilled_lora)
        self.assertEqual(args.sample_stage2_steps, len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1)
        self.assertEqual(args.sample_steps, len(DISTILLED_SIGMA_VALUES) - 1)
        self.assertEqual(args.sample_sigmas, ",".join(str(v) for v in DISTILLED_SIGMA_VALUES))
        self.assertEqual(args.guidance_scale, 1.0)
        self.assertEqual(args.cfg_scale, 1.0)

    def test_generator_merge_lora_weights_uses_runtime_helper_for_nf4_safe_apply(self):
        trainer = mock.Mock()
        trainer._apply_sampling_lora_network.return_value = (0, 1, {})
        transformer = _FakeTransformer()
        fake_network = mock.Mock()

        with mock.patch(
            "musubi_tuner.ltx2_generate_video.load_file",
            return_value={"lora_unet_model_linear.lora_down.weight": torch.randn(2, 4)},
        ), mock.patch(
            "musubi_tuner.ltx2_generate_video.lora_ltx2.create_arch_network_from_weights",
            return_value=fake_network,
        ):
            _merge_lora_weights(
                trainer,
                transformer,
                ["/tmp/example.safetensors"],
                [0.8],
                None,
                None,
            )

        trainer._apply_sampling_lora_network.assert_called_once()
        fake_network.merge_to.assert_not_called()

    def test_sample_images_applies_and_restores_sampling_lora(self):
        trainer = _SamplingHooksTrainer()
        accelerator = _FakeAccelerator()
        transformer = _FakeTransformer()
        args = Namespace(
            sample_every_n_steps=1,
            sample_every_n_epochs=None,
            sample_at_first=False,
            output_dir=tempfile.mkdtemp(),
            sample_prompts="dummy.txt",
            compile=False,
            sampling_lora_weight=["dummy.safetensors"],
            sampling_lora_multiplier=[0.8],
        )

        trainer.sample_images(
            accelerator,
            args,
            epoch=None,
            steps=1,
            vae=None,
            transformer=transformer,
            sample_parameters=[{"prompt": "hello"}],
            dit_dtype=torch.float32,
        )

        self.assertEqual(
            trainer.events,
            [("apply", "cpu"), ("sample", "hello"), ("restore", 1, "cpu")],
        )

    def test_sample_images_switches_compiled_fp8_transformer_to_eager_for_sampling_lora(self):
        trainer = _SamplingHooksTrainer()
        accelerator = _FakeAccelerator()
        transformer = _FakeCompileTransformer()
        args = Namespace(
            sample_every_n_steps=1,
            sample_every_n_epochs=None,
            sample_at_first=False,
            output_dir=tempfile.mkdtemp(),
            sample_prompts="dummy.txt",
            compile=True,
            sampling_lora_weight=["dummy.safetensors"],
            sampling_lora_multiplier=[0.8],
        )

        with mock.patch.object(trainer, "_sampling_lora_requires_eager_fallback", return_value=True), mock.patch(
            "musubi_tuner.utils.model_utils.swap_compiled_modules_with_eager",
            return_value=[("module", "forward", object())],
        ) as swap_mock, mock.patch(
            "musubi_tuner.utils.model_utils.restore_compiled_modules"
        ) as restore_mock:
            trainer.sample_images(
                accelerator,
                args,
                epoch=None,
                steps=1,
                vae=None,
                transformer=transformer,
                sample_parameters=[{"prompt": "hello"}],
                dit_dtype=torch.float32,
            )

        swap_mock.assert_called_once_with(transformer)
        restore_mock.assert_called_once()
        self.assertIn(("apply", "cpu"), trainer.events)
        self.assertIn(("restore", 1, "cpu"), trainer.events)

    def test_parse_sample_sigmas_accepts_comma_string(self):
        trainer = LTX2NetworkTrainer()
        parsed = trainer._parse_sample_sigmas_value("1.0, 0.99375, 0.5, 0.0")
        self.assertEqual(parsed, [1.0, 0.99375, 0.5, 0.0])

    def test_apply_sample_defaults_uses_custom_sigma_list_as_default(self):
        trainer = LTX2NetworkTrainer()
        args = Namespace(
            height=512,
            width=768,
            sample_num_frames=45,
            guidance_scale=1.0,
            discrete_flow_shift=5.0,
            sample_sigmas="1.0,0.9,0.5,0.0",
        )

        params = trainer._apply_sample_defaults(args, [{"prompt": "hello"}])

        self.assertEqual(params[0]["sample_sigmas"], [1.0, 0.9, 0.5, 0.0])
        self.assertEqual(params[0]["sample_steps"], 3)

    def test_apply_sample_defaults_propagates_stage_specific_distilled_lora_multipliers(self):
        trainer = LTX2NetworkTrainer()
        args = Namespace(
            height=512,
            width=768,
            sample_num_frames=45,
            guidance_scale=1.0,
            discrete_flow_shift=5.0,
            sample_sigmas=None,
            sample_stage1_distilled_lora_multiplier=0.8,
            sample_stage2_distilled_lora_multiplier=0.6,
        )

        params = trainer._apply_sample_defaults(
            args,
            [
                {"prompt": "default"},
                {
                    "prompt": "override",
                    "stage1_distilled_lora_multiplier": 0.4,
                    "stage2_distilled_lora_multiplier": 0.2,
                },
            ],
        )

        self.assertEqual(params[0]["stage1_distilled_lora_multiplier"], 0.8)
        self.assertEqual(params[0]["stage2_distilled_lora_multiplier"], 0.6)
        self.assertEqual(params[1]["stage1_distilled_lora_multiplier"], 0.4)
        self.assertEqual(params[1]["stage2_distilled_lora_multiplier"], 0.2)

    def test_resolve_sample_sigmas_uses_explicit_values(self):
        trainer = LTX2NetworkTrainer()

        sigmas, effective_steps = trainer._resolve_sample_sigmas(
            {"sample_sigmas": [1.0, 0.9, 0.5, 0.0]},
            sample_steps=20,
            device=torch.device("cpu"),
        )

        self.assertEqual(effective_steps, 3)
        self.assertEqual(sigmas.dtype, torch.float32)
        self.assertTrue(torch.equal(sigmas, torch.tensor([1.0, 0.9, 0.5, 0.0], dtype=torch.float32)))

    def test_resolve_sample_sigmas_uses_scheduler_when_not_explicit(self):
        trainer = LTX2NetworkTrainer()

        sigmas, effective_steps = trainer._resolve_sample_sigmas(
            {},
            sample_steps=4,
            device=torch.device("cpu"),
        )

        self.assertEqual(effective_steps, 4)
        self.assertEqual(sigmas.dtype, torch.float32)
        self.assertEqual(sigmas.device.type, "cpu")
        self.assertEqual(sigmas.shape[0], 5)

    def test_generator_build_prompt_list_parses_cli_sample_sigmas(self):
        trainer = LTX2NetworkTrainer()
        args = Namespace(
            sample_prompts=None,
            prompt="hello",
            negative_prompt="",
            height=512,
            width=768,
            frame_count=45,
            frame_rate=25.0,
            sample_steps=20,
            sample_sigmas="1.0,0.9,0.5,0.0",
            guidance_scale=1.0,
            discrete_flow_shift=5.0,
            seed=123,
            cfg_scale=None,
            sample_stage1_distilled_lora_multiplier=1.0,
            sample_stage2_distilled_lora_multiplier=1.0,
        )

        prompts = _build_prompt_list(trainer, args, _FakeAccelerator())

        self.assertEqual(prompts[0]["sample_sigmas"], [1.0, 0.9, 0.5, 0.0])
        self.assertEqual(prompts[0]["sample_steps"], 3)

    def test_sampling_prompt_embeds_trim_to_video_dims_when_audio_preview_disabled(self):
        trainer = LTX2NetworkTrainer()
        transformer = type(
            "FakeTransformer",
            (),
            {
                "cross_attention_dim": 4096,
                "audio_cross_attention_dim": 2048,
                "caption_projection": None,
            },
        )()
        prompt_embeds = torch.randn(2, 8, 6144)

        resolved = trainer._resolve_sampling_prompt_embeds(
            transformer=transformer,
            prompt_embeds=prompt_embeds,
            enable_audio_preview=False,
            audio_ref_only_ic_sampling=False,
        )

        self.assertEqual(resolved.shape, (2, 8, 4096))
        self.assertTrue(torch.equal(resolved, prompt_embeds[..., :4096]))

    def test_sampling_prompt_embeds_keep_full_dims_when_audio_preview_enabled(self):
        trainer = LTX2NetworkTrainer()
        transformer = type(
            "FakeTransformer",
            (),
            {
                "cross_attention_dim": 4096,
                "audio_cross_attention_dim": 2048,
                "caption_projection": _FakeCaptionProjection(4096),
            },
        )()
        prompt_embeds = torch.randn(2, 8, 6144)

        resolved = trainer._resolve_sampling_prompt_embeds(
            transformer=transformer,
            prompt_embeds=prompt_embeds,
            enable_audio_preview=True,
            audio_ref_only_ic_sampling=False,
        )

        self.assertIs(resolved, prompt_embeds)

    def test_nf4_runtime_overlay_avoids_merge_against_packed_weight_shape(self):
        trainer = LTX2NetworkTrainer()
        module = torch.nn.Linear(4, 4, bias=False)
        module._nf4_quantized = True
        module.register_buffer("scale_weight", torch.ones((4, 1, 1), dtype=torch.bfloat16))
        del module.weight
        module.register_buffer("weight", torch.zeros((4, 2), dtype=torch.uint8))
        network = type("Net", (), {"text_encoder_loras": [], "unet_loras": [_FakeSamplingLora(module)]})()
        weights_sd = {
            "lora_unet_model_linear.lora_down.weight": torch.randn(2, 4, dtype=torch.bfloat16),
            "lora_unet_model_linear.lora_up.weight": torch.randn(4, 2, dtype=torch.bfloat16),
        }

        merged, runtime_attached, backups = trainer._apply_sampling_lora_network(
            network,
            weights_sd,
            device=torch.device("cpu"),
        )

        self.assertEqual(merged, 0)
        self.assertEqual(runtime_attached, 1)
        self.assertEqual(backups, {})

    @unittest.skipIf(not hasattr(torch, "float8_e4m3fn"), "float8 dtype unavailable")
    def test_float8_runtime_overlay_uses_float32_buffers(self):
        trainer = _Float8OverlayTrainer()
        module = torch.nn.Linear(4, 4, bias=False)
        module.weight = torch.nn.Parameter(module.weight.detach().to(torch.float8_e4m3fn), requires_grad=False)
        network = type("Net", (), {"text_encoder_loras": [], "unet_loras": [_FakeSamplingLora(module)]})()
        weights_sd = {
            "lora_unet_model_linear.lora_down.weight": torch.randn(2, 4, dtype=torch.bfloat16),
            "lora_unet_model_linear.lora_up.weight": torch.randn(4, 2, dtype=torch.bfloat16),
        }

        merged, runtime_attached, backups = trainer._apply_sampling_lora_network(
            network,
            weights_sd,
            device=torch.device("cpu"),
        )

        self.assertEqual(merged, 0)
        self.assertEqual(runtime_attached, 1)
        self.assertEqual(backups, {})
        adapter = module._sampling_lora_runtime_adapters[0]
        self.assertEqual(getattr(module, adapter["A"]).dtype, torch.float32)
        self.assertEqual(getattr(module, adapter["B"]).dtype, torch.float32)

    def test_runtime_overlay_weights_follow_module_device_moves(self):
        trainer = LTX2NetworkTrainer()
        module = torch.nn.Linear(4, 4, bias=False)

        trainer._attach_sampling_lora_runtime_overlay(
            module,
            torch.randn(2, 4),
            torch.randn(4, 2),
            0.6,
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
        )

        adapter = module._sampling_lora_runtime_adapters[0]
        self.assertIsInstance(adapter["A"], str)
        self.assertIsInstance(adapter["B"], str)
        self.assertEqual(getattr(module, adapter["A"]).device.type, "cpu")
        self.assertEqual(getattr(module, adapter["B"]).device.type, "cpu")

        module.to("meta")

        self.assertEqual(getattr(module, adapter["A"]).device.type, "meta")
        self.assertEqual(getattr(module, adapter["B"]).device.type, "meta")

    def test_normalize_sampling_lora_weights_converts_comfy_ltx_keys(self):
        trainer = LTX2NetworkTrainer()
        weights_sd = {
            "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight": torch.randn(2, 4),
            "diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight": torch.randn(4, 2),
            "diffusion_model.transformer_blocks.0.attn1.to_k.alpha": torch.tensor(2.0),
            "diffusion_model.transformer_blocks.0.audio_attn2.to_out.0.lora_A.weight": torch.randn(2, 4),
            "diffusion_model.transformer_blocks.0.audio_attn2.to_out.0.lora_B.weight": torch.randn(4, 2),
            "diffusion_model.transformer_blocks.0.audio_attn2.to_out.0.alpha": torch.tensor(2.0),
        }

        normalized = trainer.normalize_sampling_lora_weights(
            Namespace(network_module="musubi_tuner.networks.lora_ltx2"),
            weights_sd,
            "dummy.safetensors",
        )

        self.assertIn(
            "lora_unet_model_transformer_blocks_0_attn1_to_k.lora_down.weight",
            normalized,
        )
        self.assertIn(
            "lora_unet_model_transformer_blocks_0_attn1_to_k.lora_up.weight",
            normalized,
        )
        self.assertIn(
            "lora_unet_model_transformer_blocks_0_attn1_to_k.alpha",
            normalized,
        )
        self.assertIn(
            "lora_unet_model_transformer_blocks_0_audio_attn2_to_out_0.lora_down.weight",
            normalized,
        )
        self.assertIn(
            "lora_unet_model_transformer_blocks_0_audio_attn2_to_out_0.lora_up.weight",
            normalized,
        )

    def test_normalize_sampling_lora_weights_uses_ltx2_default_module_when_args_field_missing(self):
        trainer = LTX2NetworkTrainer()
        weights_sd = {"lora_unet_model_dummy.lora_down.weight": torch.randn(2, 2)}

        with mock.patch.object(trainer, "convert_weight_keys", side_effect=lambda sd, network_module_name: (sd, network_module_name)) as convert:
            normalized, network_module_name = trainer.normalize_sampling_lora_weights(
                Namespace(),
                weights_sd,
                "/tmp/a.safetensors",
            )

        self.assertIs(normalized, weights_sd)
        self.assertEqual(network_module_name, "musubi_tuner.networks.lora_ltx2")
        convert.assert_called_once_with(weights_sd, "musubi_tuner.networks.lora_ltx2")

    def test_apply_sampling_lora_specs_uses_explicit_specs_and_restore_clears_them(self):
        trainer = LTX2NetworkTrainer()
        transformer = _FakeTransformer()
        args = Namespace(network_module="musubi_tuner.networks.lora_ltx2")
        network_module = mock.Mock()
        network_module.create_arch_network_from_weights.side_effect = lambda multiplier, weights_sd, unet, for_inference: {
            "multiplier": multiplier,
            "weights_sd": weights_sd,
            "unet": unet,
            "for_inference": for_inference,
        }

        with mock.patch("musubi_tuner.hv_train_network.importlib.import_module", return_value=network_module), \
             mock.patch.object(trainer, "_load_sampling_lora_weights", side_effect=[{"w": "a"}, {"w": "b"}]) as load_weights, \
             mock.patch.object(trainer, "_apply_sampling_lora_network", side_effect=[(0, 1, {}), (1, 0, {123: ("module", "backup")})]) as apply_network, \
             mock.patch.object(trainer, "_clear_sampling_lora_runtime_overlays", side_effect=[0, 1]) as clear_runtime, \
             mock.patch.object(trainer, "_restore_sampling_lora_network", return_value=1) as restore_network:
            applied = trainer._apply_sampling_lora_specs(
                args,
                transformer,
                torch.device("cpu"),
                specs=[("/tmp/a.safetensors", 0.5), ("/tmp/b.safetensors", 1.0)],
            )
            trainer._restore_sampling_lora_specs(
                args,
                transformer,
                torch.device("cpu"),
                applied,
            )

        self.assertEqual(
            load_weights.call_args_list,
            [mock.call(args, "/tmp/a.safetensors"), mock.call(args, "/tmp/b.safetensors")],
        )
        self.assertEqual(network_module.create_arch_network_from_weights.call_count, 2)
        self.assertEqual(apply_network.call_count, 2)
        self.assertEqual(applied, [("/tmp/a.safetensors", {}), ("/tmp/b.safetensors", {123: ("module", "backup")})])
        self.assertEqual(clear_runtime.call_count, 2)
        restore_network.assert_called_once_with({123: ("module", "backup")})

    def test_apply_sampling_lora_specs_uses_ltx2_default_module_when_args_field_missing(self):
        trainer = LTX2NetworkTrainer()
        transformer = _FakeTransformer()
        network_module = mock.Mock()
        network_module.create_arch_network_from_weights.side_effect = lambda multiplier, weights_sd, unet, for_inference: {
            "multiplier": multiplier,
            "weights_sd": weights_sd,
            "unet": unet,
            "for_inference": for_inference,
        }

        with mock.patch("musubi_tuner.hv_train_network.importlib.import_module", return_value=network_module) as import_module, \
             mock.patch.object(trainer, "_load_sampling_lora_weights", return_value={"w": "a"}) as load_weights, \
             mock.patch.object(trainer, "_apply_sampling_lora_network", return_value=(0, 1, {})) as apply_network:
            applied = trainer._apply_sampling_lora_specs(
                Namespace(),
                transformer,
                torch.device("cpu"),
                specs=[("/tmp/a.safetensors", 0.5)],
            )

        import_module.assert_called_once_with("musubi_tuner.networks.lora_ltx2")
        load_weights.assert_called_once()
        apply_network.assert_called_once()
        self.assertEqual(applied, [("/tmp/a.safetensors", {})])

    def test_resolve_standalone_autocache_dir_uses_cwd_default(self):
        trainer = LTX2NetworkTrainer()

        with mock.patch("musubi_tuner.ltx2_train_network.os.getcwd", return_value="/tmp/run"):
            cache_dir = trainer._resolve_standalone_autocache_dir(Namespace(autocache="__cwd__"))

        self.assertEqual(cache_dir, "/tmp/run/.cache/ltx2_prompt_embeddings")

    def test_prepare_sample_prompt_embeddings_batch_uses_autocache_without_loading_gemma_when_all_cached(self):
        trainer = _AutocacheSamplingTrainer()
        accelerator = _FakeAccelerator()
        prompt_embeds = torch.randn(3, 4)
        prompt_mask = torch.ones(3, dtype=torch.bool)
        neg_embeds = torch.randn(2, 4)
        neg_mask = torch.ones(2, dtype=torch.bool)
        sample_parameter = {
            "prompt": "Prompt A",
            "negative_prompt": "Negative A",
            "guidance_scale": 3.0,
        }

        with mock.patch.object(trainer, "_resolve_standalone_autocache_dir", return_value="/tmp/cache"), \
             mock.patch.object(
                 trainer,
                 "_load_standalone_prompt_embedding_cache",
                 side_effect=[(prompt_embeds, prompt_mask), (neg_embeds, neg_mask)],
             ) as load_cache, \
             mock.patch.object(trainer, "_build_text_encoder") as build_text_encoder, \
             mock.patch.object(trainer, "_encode_prompt_text") as encode_prompt_text, \
             mock.patch.object(trainer, "_cleanup_text_encoder") as cleanup_text_encoder:
            trainer._prepare_sample_prompt_embeddings_batch(
                accelerator,
                Namespace(autocache="__cwd__", use_precached_sample_prompts=False, precache_sample_prompts=False),
                [sample_parameter],
            )

        self.assertTrue(torch.equal(sample_parameter["prompt_embeds"], prompt_embeds))
        self.assertTrue(torch.equal(sample_parameter["prompt_attention_mask"], prompt_mask))
        self.assertTrue(torch.equal(sample_parameter["negative_prompt_embeds"], neg_embeds))
        self.assertTrue(torch.equal(sample_parameter["negative_prompt_attention_mask"], neg_mask))
        self.assertEqual(load_cache.call_count, 2)
        build_text_encoder.assert_not_called()
        encode_prompt_text.assert_not_called()
        cleanup_text_encoder.assert_not_called()

    def test_prepare_sample_prompt_embeddings_batch_saves_missing_autocache_entries_after_encoding(self):
        trainer = _AutocacheSamplingTrainer()
        accelerator = _FakeAccelerator()
        prompt_embeds = torch.randn(3, 4)
        prompt_mask = torch.ones(3, dtype=torch.bool)
        sample_parameter = {
            "prompt": "Prompt A",
            "guidance_scale": 1.0,
        }

        with mock.patch.object(trainer, "_resolve_standalone_autocache_dir", return_value="/tmp/cache"), \
             mock.patch.object(trainer, "_load_standalone_prompt_embedding_cache", return_value=None) as load_cache, \
             mock.patch.object(trainer, "_build_text_encoder", return_value=torch.bfloat16) as build_text_encoder, \
             mock.patch.object(trainer, "_encode_prompt_text", return_value=(prompt_embeds, prompt_mask)) as encode_prompt_text, \
             mock.patch.object(trainer, "_save_standalone_prompt_embedding_cache") as save_cache, \
             mock.patch.object(trainer, "_cleanup_text_encoder") as cleanup_text_encoder:
            trainer._prepare_sample_prompt_embeddings_batch(
                accelerator,
                Namespace(autocache="/tmp/cache", use_precached_sample_prompts=False, precache_sample_prompts=False),
                [sample_parameter],
            )

        self.assertEqual(load_cache.call_count, 1)
        build_text_encoder.assert_called_once()
        encode_prompt_text.assert_called_once_with(accelerator, "Prompt A", torch.bfloat16)
        save_cache.assert_called_once()
        cleanup_text_encoder.assert_called_once_with(accelerator)
        self.assertTrue(torch.equal(sample_parameter["prompt_embeds"], prompt_embeds))
        self.assertTrue(torch.equal(sample_parameter["prompt_attention_mask"], prompt_mask))

    def test_main_cache_prompt_file_warms_cache_without_loading_transformer(self):
        fake_accelerator = _FakeAccelerator()
        trainer = mock.Mock()
        trainer.blocks_to_swap = 0
        trainer.dit_dtype = torch.bfloat16

        with mock.patch.object(
            sys,
            "argv",
            [
                "ltx2_generate_video.py",
                "--ltx2_checkpoint",
                "/tmp/ltx.safetensors",
                "--gemma_root",
                "/tmp/gemma",
                "--prompt",
                "hello",
                "--cache_prompt_file",
            ],
        ), mock.patch("musubi_tuner.ltx2_generate_video.Accelerator", return_value=fake_accelerator), \
             mock.patch("musubi_tuner.ltx2_generate_video.LTX2NetworkTrainer", return_value=trainer), \
             mock.patch("musubi_tuner.ltx2_generate_video._build_prompt_list", return_value=[{"prompt": "hello"}]) as build_prompt_list, \
             mock.patch("musubi_tuner.ltx2_generate_video.os.makedirs"), \
             mock.patch("musubi_tuner.ltx2_generate_video.clean_memory_on_device"):
            main()

        trainer.handle_model_specific_args.assert_called_once()
        build_prompt_list.assert_called_once()
        trainer._prepare_sample_prompt_embeddings_batch.assert_called_once()
        trainer.load_transformer.assert_not_called()
        trainer.sample_images.assert_not_called()

    def test_should_merge_cli_loras_once_skips_prompt_file_mode(self):
        args = Namespace(
            lora_weight=["/tmp/a.safetensors"],
            sample_prompts="/tmp/prompts.toml",
            prompt=None,
        )

        self.assertFalse(_should_merge_cli_loras_once(args))

    def test_should_merge_cli_loras_once_keeps_single_prompt_mode(self):
        args = Namespace(
            lora_weight=["/tmp/a.safetensors"],
            sample_prompts=None,
            prompt="hello world",
        )

        self.assertTrue(_should_merge_cli_loras_once(args))

    def test_sample_images_applies_prompt_local_loras_per_sample_without_leakage(self):
        trainer = _PromptLocalSamplingTrainer()
        accelerator = _FakeAccelerator()
        transformer = _FakeTransformer()
        args = Namespace(
            sample_at_first=True,
            sample_every_n_steps=None,
            sample_every_n_epochs=None,
            sample_prompts="dummy.toml",
            output_dir=tempfile.mkdtemp(prefix="ltx2-prompt-local-"),
            sample_with_offloading=False,
            use_precached_sample_prompts=False,
            precache_sample_prompts=False,
            sample_audio_subprocess=True,
            sample_disable_audio=False,
            sample_audio_only=False,
            ltx_mode="video",
            vae_dtype=None,
            sampling_lora_weight=[],
            sampling_lora_multiplier=[],
            compile=False,
        )
        sample_parameters = [
            {
                "prompt": "Prompt A",
                "enum": 0,
                "resolved_loras": [
                    {"path": "/a.safetensors", "weight": 0.5, "merge": False},
                    {"path": "/b.safetensors", "weight": 0.8, "merge": False},
                ],
            },
            {
                "prompt": "Prompt B",
                "enum": 1,
                "resolved_loras": [{"path": "/c.safetensors", "weight": 0.9, "merge": False}],
            },
            {
                "prompt": "Prompt C",
                "enum": 2,
                "resolved_loras": [],
            },
        ]

        with mock.patch("musubi_tuner.ltx2_train_network.PartialState", _FakePartialState), mock.patch(
            "musubi_tuner.ltx2_train_network.clean_memory_on_device", lambda device: None
        ):
            trainer.sample_images(
                accelerator=accelerator,
                args=args,
                epoch=0,
                steps=0,
                vae=object(),
                transformer=transformer,
                sample_parameters=sample_parameters,
                dit_dtype=torch.bfloat16,
            )

        self.assertEqual(
            trainer.events,
            [
                ("apply_specs", (("/a.safetensors", 0.5), ("/b.safetensors", 0.8)), "cpu"),
                ("sample", "Prompt A"),
                ("restore_specs", (("prompt-local", {"backups": 2}),), "cpu"),
                ("apply_specs", (("/c.safetensors", 0.9),), "cpu"),
                ("sample", "Prompt B"),
                ("restore_specs", (("prompt-local", {"backups": 1}),), "cpu"),
                ("sample", "Prompt C"),
            ],
        )

    def test_sample_images_live_reload_updates_prompt_local_loras_from_toml(self):
        trainer = _PromptLocalSamplingTrainer()
        accelerator = _FakeAccelerator()
        transformer = _FakeTransformer()
        prompt_path = Path(tempfile.mkdtemp(prefix="ltx2-live-reload-")) / "prompts.toml"
        prompt_path.write_text(
            """
[prompt]

[[prompt.subset]]
prompt = "Prompt A"
loras = [{ path = "/fresh.safetensors", weight = 0.75, merge = false }]
""".strip(),
            encoding="utf-8",
        )
        args = Namespace(
            sample_at_first=True,
            sample_every_n_steps=None,
            sample_every_n_epochs=None,
            sample_prompts=str(prompt_path),
            output_dir=tempfile.mkdtemp(prefix="ltx2-prompt-live-reload-"),
            sample_with_offloading=False,
            use_precached_sample_prompts=False,
            precache_sample_prompts=False,
            sample_audio_subprocess=True,
            sample_disable_audio=False,
            sample_audio_only=False,
            sample_live_reload_loras=True,
            ltx_mode="video",
            vae_dtype=None,
            sampling_lora_weight=[],
            sampling_lora_multiplier=[],
            compile=False,
        )
        prompt_embeds = torch.randn(1, 2, 4)
        sample_parameters = [
            {
                "prompt": "Prompt A",
                "enum": 0,
                "prompt_embeds": prompt_embeds,
                "prompt_attention_mask": torch.ones(1, 2, dtype=torch.int64),
                "resolved_loras": [{"path": "/stale.safetensors", "weight": 0.25, "merge": False}],
            }
        ]

        with mock.patch("musubi_tuner.ltx2_train_network.PartialState", _FakePartialState), mock.patch(
            "musubi_tuner.ltx2_train_network.clean_memory_on_device", lambda device: None
        ):
            trainer.sample_images(
                accelerator=accelerator,
                args=args,
                epoch=0,
                steps=0,
                vae=object(),
                transformer=transformer,
                sample_parameters=sample_parameters,
                dit_dtype=torch.bfloat16,
            )

        self.assertEqual(
            trainer.events,
            [
                ("apply_specs", (("/fresh.safetensors", 0.75),), "cpu"),
                ("sample", "Prompt A"),
                ("restore_specs", (("prompt-local", {"backups": 1}),), "cpu"),
            ],
        )
        self.assertIs(sample_parameters[0]["prompt_embeds"], prompt_embeds)

    def test_refresh_live_sampling_loras_clears_sampling_lora_cache_when_enabled(self):
        trainer = LTX2NetworkTrainer()
        trainer._sampling_lora_cache["/tmp/example.safetensors"] = {"w": torch.tensor(1.0)}
        args = Namespace(
            sample_live_reload_loras=True,
            sample_prompts="dummy.txt",
            sampling_lora_weight=[],
            sampling_lora_multiplier=[],
        )

        trainer._refresh_live_sampling_loras(args, sample_parameters=[{"prompt": "Prompt A", "enum": 0}])

        self.assertEqual(trainer._sampling_lora_cache, {})

    def test_sample_images_temporarily_applies_sampling_lora(self):
        trainer = _RecordingTrainer()
        accelerator = _FakeAccelerator()
        transformer = _FakeTransformer()
        args = Namespace(
            sample_at_first=True,
            sample_every_n_steps=None,
            sample_every_n_epochs=None,
            sample_prompts="dummy.txt",
            output_dir=tempfile.mkdtemp(prefix="ltx2-sample-lora-"),
            sample_with_offloading=False,
            use_precached_sample_prompts=False,
            precache_sample_prompts=False,
            sample_audio_subprocess=True,
            sample_disable_audio=False,
            sample_audio_only=False,
            ltx_mode="video",
            vae_dtype=None,
            sampling_lora_weight=["/tmp/sample-lora.safetensors"],
            sampling_lora_multiplier=[0.8],
            compile=False,
        )
        sample_parameters = [{"prompt": "hello world", "enum": 0}]

        with mock.patch("musubi_tuner.ltx2_train_network.PartialState", _FakePartialState), mock.patch(
            "musubi_tuner.ltx2_train_network.clean_memory_on_device", lambda device: None
        ):
            trainer.sample_images(
                accelerator=accelerator,
                args=args,
                epoch=0,
                steps=0,
                vae=object(),
                transformer=transformer,
                sample_parameters=sample_parameters,
                dit_dtype=torch.bfloat16,
            )

        self.assertEqual(trainer.applied, [(("/tmp/sample-lora.safetensors",), "cpu")])
        self.assertEqual(trainer.restored, [(("/tmp/sample-lora.safetensors",), 1, "cpu")])
        self.assertEqual(trainer.sampled, ["hello world"])
        self.assertEqual(transformer.inference_switches, 1)
        self.assertEqual(transformer.training_switches, 1)

    def test_resolve_step_sampling_requests_moves_sample_at_first_to_step_one(self):
        args = Namespace(
            sample_at_first=True,
            sample_every_n_steps=None,
            sample_every_n_epochs=None,
        )

        initial, regular = resolve_step_sampling_requests(
            args,
            global_step=1,
            epoch=None,
            initial_sample_pending=True,
        )

        self.assertTrue(initial)
        self.assertFalse(regular)

    def test_resolve_step_sampling_requests_dedupes_initial_and_regular_step_one_sample(self):
        args = Namespace(
            sample_at_first=True,
            sample_every_n_steps=1,
            sample_every_n_epochs=None,
        )

        initial, regular = resolve_step_sampling_requests(
            args,
            global_step=1,
            epoch=None,
            initial_sample_pending=True,
        )

        self.assertTrue(initial)
        self.assertFalse(regular)

    def test_sample_images_force_sample_bypasses_regular_schedule_gate(self):
        trainer = _RecordingTrainer()
        accelerator = _FakeAccelerator()
        transformer = _FakeTransformer()
        args = Namespace(
            sample_at_first=True,
            sample_every_n_steps=None,
            sample_every_n_epochs=None,
            sample_prompts="dummy.txt",
            output_dir=tempfile.mkdtemp(prefix="ltx2-force-sample-"),
            sample_with_offloading=False,
            use_precached_sample_prompts=False,
            precache_sample_prompts=False,
            sample_audio_subprocess=True,
            sample_disable_audio=False,
            sample_audio_only=False,
            ltx_mode="video",
            vae_dtype=None,
            sampling_lora_weight=[],
            sampling_lora_multiplier=[],
            compile=False,
        )
        sample_parameters = [{"prompt": "hello delayed first sample", "enum": 0}]

        with mock.patch("musubi_tuner.ltx2_train_network.PartialState", _FakePartialState), mock.patch(
            "musubi_tuner.ltx2_train_network.clean_memory_on_device", lambda device: None
        ):
            trainer.sample_images(
                accelerator=accelerator,
                args=args,
                epoch=None,
                steps=1,
                vae=object(),
                transformer=transformer,
                sample_parameters=sample_parameters,
                dit_dtype=torch.bfloat16,
                force_sample=True,
            )

        self.assertEqual(trainer.sampled, ["hello delayed first sample"])
        self.assertEqual(transformer.inference_switches, 1)
        self.assertEqual(transformer.training_switches, 1)


if __name__ == "__main__":
    unittest.main()
