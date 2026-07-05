"""LoRA training entry point for Boogu Image Base."""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator, init_empty_weights
from diffusers.utils.torch_utils import randn_tensor

from musubi_tuner.boogu_image.boogu_utils import (
    BOOGU_PATCH_SIZE,
    BOOGU_VAE_SCALE_FACTOR,
    BOOGU_VAE_SCALING_FACTOR,
    BOOGU_VAE_SHIFT_FACTOR,
    load_boogu_autoencoder_kl,
)
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_BOOGU_IMAGE, ARCHITECTURE_BOOGU_IMAGE_FULL
from musubi_tuner.hv_train_network import NetworkTrainer, clean_memory_on_device, load_prompts, read_config_from_file, setup_parser_common
from musubi_tuner.utils import model_utils


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


BOOGU_FP8_OPTIMIZATION_TARGET_KEYS = [
    "noise_refiner.",
    "ref_image_refiner.",
    "context_refiner.",
    "double_stream_layers.",
    "single_stream_layers.",
]
BOOGU_FP8_OPTIMIZATION_EXCLUDE_KEYS = [
    "norm",
    "time_caption_embed",
    "x_embedder",
    "ref_image_patch_embedder",
    "image_index_embedding",
    "norm_out",
]
BOOGU_SAMPLE_QWEN_IMAGE_AREA = 384 * 384
BOOGU_SAMPLE_REF_IMAGE_AREA = 1024 * 1024
BOOGU_SAMPLE_REF_IMAGE_ALIGN = 16
BOOGU_DMD_DEFAULT_CONDITIONING_SIGMA = 0.001
BOOGU_FLOW_SAMPLER_NAMES = {"", "default", "flow"}
BOOGU_DMD_SAMPLER_NAMES = {"dmd", "dmd_turbo", "turbo"}


def _normalize_boogu_sampler(value) -> str:
    sampler = str(value or "flow").strip().lower().replace("-", "_")
    if sampler in BOOGU_FLOW_SAMPLER_NAMES:
        return "flow"
    if sampler in BOOGU_DMD_SAMPLER_NAMES:
        return "dmd"
    raise ValueError(f"Unsupported Boogu sample sampler {value!r}; expected 'flow' or 'dmd'.")


def _resize_pil_image_to_area(image, target_area: int, align_to: int | None = None):
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Boogu sample input image has invalid dimensions.")
    scale = math.sqrt(float(target_area) / float(width * height))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    if align_to is not None:
        resized_width = max(align_to, round(resized_width / align_to) * align_to)
        resized_height = max(align_to, round(resized_height / align_to) * align_to)
    if (resized_width, resized_height) == image.size:
        return image
    from PIL import Image

    return image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)


def load_boogu_sample_pil_image(image_path: str, target_area: int, align_to: int | None = None):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    return _resize_pil_image_to_area(image, target_area=target_area, align_to=align_to)


def load_boogu_sample_ref_image_tensor(image_path: str) -> torch.Tensor:
    import numpy as np

    image = load_boogu_sample_pil_image(
        image_path,
        target_area=BOOGU_SAMPLE_REF_IMAGE_AREA,
        align_to=BOOGU_SAMPLE_REF_IMAGE_ALIGN,
    )
    array = np.asarray(image, dtype=np.uint8)
    tensor = torch.from_numpy(array[..., :3].copy()).permute(2, 0, 1).to(dtype=torch.float32)
    return tensor / 127.5 - 1.0


def resolve_boogu_sample_input_image_path(prompt_dict: dict, prompt_file: str | os.PathLike | None = None) -> str | None:
    control_value = prompt_dict.get("control_image_path")
    image_value = prompt_dict.get("input_image")
    has_input_image = image_value is not None and image_value != ""
    has_control_image = control_value is not None and control_value != "" and control_value != []
    if has_input_image and has_control_image:
        raise ValueError("Boogu sample prompts support only one edit input image.")
    if image_value is None or image_value == "":
        image_value = control_value
    if image_value is None or image_value == "":
        return None
    if isinstance(image_value, list):
        if len(image_value) == 0:
            return None
        if len(image_value) != 1:
            raise ValueError("Boogu sample prompts support only one edit input image.")
        image_value = image_value[0]
    if not isinstance(image_value, str):
        raise ValueError("Boogu sample prompt input image must be a single image path string.")

    image_path = Path(os.path.expanduser(image_value))
    if not image_path.is_absolute() and prompt_file is not None:
        image_path = Path(prompt_file).expanduser().resolve().parent / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Boogu sample input image does not exist: {image_path}")
    return str(image_path)


class BooguImageNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_frame_stride = 1
        self._i2v_training = False
        self._control_training = False
        self._freqs_cis = None

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_BOOGU_IMAGE

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_BOOGU_IMAGE_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.bfloat16 if getattr(args, "mixed_precision", None) == "bf16" else torch.float32
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 4.0
        self.default_discrete_flow_shift = 3.0

        if getattr(args, "fp8_base", False) and not getattr(args, "fp8_scaled", False):
            raise ValueError("Boogu Image fp8 supports only scaled fp8: pass --fp8_scaled together with --fp8_base.")

        if int(getattr(args, "blocks_to_swap", 0) or 0) > 0:
            raise ValueError("Boogu Image block swap is not wired yet; use --fp8_base --fp8_scaled and gradient checkpointing first.")

        dit_path = str(getattr(args, "dit", "") or "").lower()
        if "fp8" in dit_path and dit_path.endswith(".bin"):
            raise ValueError(
                "Boogu Image direct torchao fp8 .bin training is unsupported; use bf16 safetensors and optional scaled fp8."
            )

    def _default_sampling_lora_network_module_name(self) -> Optional[str]:
        return "musubi_tuner.networks.lora_boogu_image"

    def _get_freqs_cis(self, transformer):
        if self._freqs_cis is None:
            from musubi_tuner.boogu_image.rope import get_freqs_cis

            cfg = transformer.config
            self._freqs_cis = get_freqs_cis(cfg.axes_dim_rope, cfg.axes_lens, theta=10000)
        return self._freqs_cis

    @staticmethod
    def _vae_config_value(vae, key: str, default):
        config = getattr(vae, "config", None)
        if config is None:
            return default
        if isinstance(config, dict):
            value = config.get(key, default)
        else:
            value = getattr(config, key, default)
        return default if value is None else value

    @staticmethod
    def _attention_backend_from_args(attn_mode: str) -> str:
        if attn_mode == "flash":
            return "flash"
        if attn_mode in {"torch", "sdpa", "xformers"}:
            return "native"
        raise ValueError(f"Boogu Image supports --sdpa or --flash-attn attention, got mode {attn_mode!r}.")

    def _encode_sample_ref_latent(self, vae, image_path: str, device: torch.device) -> torch.Tensor:
        image_tensor = load_boogu_sample_ref_image_tensor(image_path).unsqueeze(0)
        vae_device = getattr(vae, "device", device)
        vae_dtype = getattr(vae, "dtype", torch.float32)
        with torch.no_grad():
            encoded = vae.encode(image_tensor.to(device=vae_device, dtype=vae_dtype)).latent_dist.sample()
            scale = float(self._vae_config_value(vae, "scaling_factor", BOOGU_VAE_SCALING_FACTOR))
            shift = float(self._vae_config_value(vae, "shift_factor", BOOGU_VAE_SHIFT_FACTOR))
            latents = (encoded - shift) * scale
        return latents[0].detach().cpu()

    def process_sample_prompts(self, args: argparse.Namespace, accelerator: Accelerator, sample_prompts: str):
        assert args.text_encoder is not None, "--text_encoder is required for Boogu sample generation during training"

        from musubi_tuner.boogu_image_cache_text_encoder_outputs import (
            build_boogu_drop_messages,
            build_boogu_edit_messages,
            build_boogu_t2i_messages,
            load_boogu_text_encoder,
        )

        device = accelerator.device
        prompts = load_prompts(sample_prompts)
        for prompt_dict in prompts:
            prompt_dict["boogu_sample_input_image_path"] = resolve_boogu_sample_input_image_path(prompt_dict, sample_prompts)

        dtype = torch.float8_e4m3fn if getattr(args, "fp8_llm", False) else torch.bfloat16
        processor, text_encoder = load_boogu_text_encoder(
            args.text_encoder,
            dtype=dtype,
            device=device,
            subfolder=getattr(args, "text_encoder_subfolder", "mllm"),
            processor_path=getattr(args, "processor", None),
        )
        text_encoder.eval()

        max_length = int(getattr(args, "max_text_length", 1024) or 1024)
        text_features: dict[tuple[str, str, str | None], torch.Tensor] = {}
        qwen_image_cache = {}
        logger.info(f"Encoding Boogu sample prompts with Qwen3-VL: {sample_prompts}")

        def encode_feature(text_encoder_model, cache_key: tuple[str, str, str | None], messages: list[dict]) -> None:
            if cache_key in text_features:
                return
            inputs = processor.apply_chat_template(
                [messages],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=False,
                truncation=True,
                max_length=max_length,
            )
            inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            output = text_encoder_model(**inputs)
            valid_len = int(inputs["attention_mask"][0].to(dtype=torch.bool).sum().item())
            text_features[cache_key] = output.last_hidden_state[0, :valid_len].to(torch.bfloat16).cpu()

        with torch.no_grad():
            for prompt_dict in prompts:
                if "negative_prompt" not in prompt_dict:
                    prompt_dict["negative_prompt"] = ""
                prompt = prompt_dict.get("prompt", "") or ""
                input_image_path = prompt_dict.get("boogu_sample_input_image_path")
                if input_image_path is not None:
                    if input_image_path not in qwen_image_cache:
                        qwen_image_cache[input_image_path] = load_boogu_sample_pil_image(
                            input_image_path,
                            target_area=BOOGU_SAMPLE_QWEN_IMAGE_AREA,
                        )
                    prompt_key = ("edit", prompt, input_image_path)
                    encode_feature(text_encoder, prompt_key, build_boogu_edit_messages(prompt, qwen_image_cache[input_image_path]))
                else:
                    prompt_key = ("t2i", prompt, None)
                    encode_feature(text_encoder, prompt_key, build_boogu_t2i_messages(prompt))
                prompt_dict["_boogu_instruction_feature_key"] = prompt_key

                negative_prompt = prompt_dict.get("negative_prompt", "") or ""
                if input_image_path is not None and not negative_prompt.strip():
                    negative_key = ("drop", "", None)
                    encode_feature(text_encoder, negative_key, build_boogu_drop_messages(""))
                else:
                    negative_key = ("t2i", negative_prompt, None)
                    encode_feature(text_encoder, negative_key, build_boogu_t2i_messages(negative_prompt))
                prompt_dict["_negative_boogu_instruction_feature_key"] = negative_key

        del text_encoder
        clean_memory_on_device(device)

        ref_latent_cache: dict[str, torch.Tensor] = {}
        sample_input_paths = [
            prompt_dict["boogu_sample_input_image_path"]
            for prompt_dict in prompts
            if prompt_dict.get("boogu_sample_input_image_path") is not None
        ]
        unique_sample_input_paths = list(dict.fromkeys(sample_input_paths))
        if unique_sample_input_paths:
            if not getattr(args, "vae", None):
                raise ValueError("Boogu sample input images require --vae so reference latents can be pre-cached.")
            vae_dtype = torch.bfloat16 if getattr(args, "vae_dtype", None) is None else model_utils.str_to_dtype(args.vae_dtype)
            vae = self.load_vae(args, vae_dtype=vae_dtype, vae_path=args.vae)
            vae.to(device=device, dtype=vae_dtype)
            vae.eval()
            try:
                for input_image_path in unique_sample_input_paths:
                    ref_latent_cache[input_image_path] = self._encode_sample_ref_latent(vae, input_image_path, device)
            finally:
                vae.to("cpu")
                del vae
                clean_memory_on_device(device)

        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            prompt_dict_copy["boogu_instruction_embed"] = text_features[prompt_dict["_boogu_instruction_feature_key"]]
            prompt_dict_copy["negative_boogu_instruction_embed"] = text_features[prompt_dict["_negative_boogu_instruction_feature_key"]]
            input_image_path = prompt_dict.get("boogu_sample_input_image_path")
            if input_image_path is not None:
                prompt_dict_copy["boogu_ref_image_hidden_states"] = [ref_latent_cache[input_image_path]]
            prompt_dict_copy.pop("_boogu_instruction_feature_key", None)
            prompt_dict_copy.pop("_negative_boogu_instruction_feature_key", None)
            sample_parameters.append(prompt_dict_copy)
        return sample_parameters

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        from musubi_tuner.boogu_image.boogu_utils import boogu_time_schedule, pad_instruction_features

        device = accelerator.device
        model = accelerator.unwrap_model(transformer)
        patch = int(getattr(model.config, "patch_size", BOOGU_PATCH_SIZE))
        in_channels = int(getattr(model.config, "in_channels", 16))
        align = BOOGU_VAE_SCALE_FACTOR * patch
        width = max(align, (int(width) // align) * align)
        height = max(align, (int(height) // align) * align)
        lat_h = height // BOOGU_VAE_SCALE_FACTOR
        lat_w = width // BOOGU_VAE_SCALE_FACTOR
        num_patch_tokens = (lat_h // patch) * (lat_w // patch)

        latents = randn_tensor(
            (1, in_channels, lat_h, lat_w),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        cond_feats, cond_mask = pad_instruction_features(
            [sample_parameter["boogu_instruction_embed"]],
            device=device,
            dtype=dit_dtype,
        )
        cfg = float(cfg_scale if cfg_scale is not None else guidance_scale)
        do_cfg = bool(do_classifier_free_guidance and cfg > 1.0)
        if do_cfg:
            uncond_feats, uncond_mask = pad_instruction_features(
                [sample_parameter["negative_boogu_instruction_embed"]],
                device=device,
                dtype=dit_dtype,
            )
        cached_ref_image_hidden_states = sample_parameter.get("boogu_ref_image_hidden_states")
        ref_image_hidden_states = None
        if cached_ref_image_hidden_states:
            ref_image_hidden_states = [
                [ref_latent.to(device=device, dtype=dit_dtype) for ref_latent in cached_ref_image_hidden_states]
            ]

        freqs_cis = self._get_freqs_cis(model)
        sampler = _normalize_boogu_sampler(sample_parameter.get("boogu_sampler"))
        if sampler == "dmd":
            if cfg != 1.0:
                raise ValueError(
                    "Boogu DMD/turbo preview sampling requires cfg_scale=1.0 "
                    "(or guidance_scale=1.0 when cfg_scale is omitted)."
                )
            if sample_steps < 1:
                raise ValueError("Boogu DMD/turbo preview sampling requires sample_steps >= 1.")
            conditioning_sigma = float(
                sample_parameter.get(
                    "boogu_dmd_conditioning_sigma",
                    sample_parameter.get("dmd_conditioning_sigma", BOOGU_DMD_DEFAULT_CONDITIONING_SIGMA),
                )
            )
            if not 0.0 <= conditioning_sigma <= 1.0:
                raise ValueError("Boogu DMD conditioning sigma must be between 0.0 and 1.0.")
            logger.info(
                "Boogu sample sampler: DMD/turbo (steps=%s, conditioning_sigma=%s)",
                sample_steps,
                conditioning_sigma,
            )
            sigmas = torch.linspace(
                conditioning_sigma,
                1.0,
                int(sample_steps) + 1,
                device=device,
                dtype=torch.float32,
            )[:-1]
            for i, sigma in enumerate(sigmas):
                sigma_value = float(sigma.item())
                boogu_t = torch.full((latents.shape[0],), sigma_value, device=device, dtype=torch.float32)
                latent_model_input = latents.to(device=device, dtype=dit_dtype)
                with torch.no_grad(), accelerator.autocast():
                    model_pred = model(
                        hidden_states=latent_model_input,
                        timestep=boogu_t,
                        instruction_hidden_states=cond_feats,
                        freqs_cis=freqs_cis,
                        instruction_attention_mask=cond_mask,
                        ref_image_hidden_states=ref_image_hidden_states,
                        return_dict=False,
                    )
                sigma_expanded = torch.full(
                    (latents.shape[0], 1, 1, 1),
                    sigma_value,
                    device=device,
                    dtype=torch.float32,
                )
                latents = latents + (1.0 - sigma_expanded) * model_pred.to(torch.float32)
                if i < len(sigmas) - 1:
                    next_sigma = float(sigmas[i + 1].item())
                    noise = randn_tensor(
                        latents.shape,
                        generator=generator,
                        device=device,
                        dtype=torch.float32,
                    )
                    next_sigma_expanded = torch.full(
                        (latents.shape[0], 1, 1, 1),
                        next_sigma,
                        device=device,
                        dtype=torch.float32,
                    )
                    latents = (1.0 - next_sigma_expanded) * noise + next_sigma_expanded * latents
        else:
            times = boogu_time_schedule(sample_steps, num_patch_tokens, device=device)
            for t, t_next in zip(times[:-1], times[1:]):
                boogu_t = t.expand(latents.shape[0]).to(device=device, dtype=torch.float32)
                latent_model_input = latents.to(device=device, dtype=dit_dtype)
                with torch.no_grad(), accelerator.autocast():
                    v_cond = model(
                        hidden_states=latent_model_input,
                        timestep=boogu_t,
                        instruction_hidden_states=cond_feats,
                        freqs_cis=freqs_cis,
                        instruction_attention_mask=cond_mask,
                        ref_image_hidden_states=ref_image_hidden_states,
                        return_dict=False,
                    )
                    if do_cfg:
                        v_uncond = model(
                            hidden_states=latent_model_input,
                            timestep=boogu_t,
                            instruction_hidden_states=uncond_feats,
                            freqs_cis=freqs_cis,
                            instruction_attention_mask=uncond_mask,
                            ref_image_hidden_states=ref_image_hidden_states,
                            return_dict=False,
                        )
                        velocity = v_uncond + cfg * (v_cond - v_uncond)
                    else:
                        velocity = v_cond
                latents = latents + velocity.to(torch.float32) * (t_next - t)

        vae.to(device)
        vae.eval()
        try:
            scale = float(self._vae_config_value(vae, "scaling_factor", BOOGU_VAE_SCALING_FACTOR))
            shift = float(self._vae_config_value(vae, "shift_factor", BOOGU_VAE_SHIFT_FACTOR))
            decode_latents = latents.to(device=device, dtype=vae.dtype) / scale + shift
            with torch.no_grad():
                decoded = vae.decode(decode_latents).sample
        finally:
            vae.to("cpu")
        clean_memory_on_device(device)

        pixels = decoded.to(torch.float32).cpu()
        pixels = (pixels / 2 + 0.5).clamp(0, 1)
        return pixels.unsqueeze(2)

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_path = args.vae or vae_path
        logger.info(f"Loading Boogu-compatible AutoencoderKL from {vae_path}")
        vae = load_boogu_autoencoder_kl(vae_path, vae_dtype)
        vae.to("cpu", dtype=vae_dtype)
        vae.eval()
        return vae

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        from musubi_tuner.boogu_image.transformer import BooguImageTransformer2DModel

        attention_backend = self._attention_backend_from_args(attn_mode)
        fp8_scaled = bool(getattr(args, "fp8_scaled", False))
        loading_device = torch.device(loading_device)

        if os.path.isfile(dit_path):
            from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
            from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
            from musubi_tuner.utils.safetensors_utils import load_safetensors

            with init_empty_weights():
                model = BooguImageTransformer2DModel()
                if dit_weight_dtype is not None:
                    model.to(dit_weight_dtype)

            if fp8_scaled:
                sd = load_safetensors_with_lora_and_fp8(
                    model_files=dit_path,
                    lora_weights_list=None,
                    lora_multipliers=None,
                    fp8_optimization=True,
                    calc_device=accelerator.device,
                    move_to_device=(loading_device == accelerator.device),
                    dit_weight_dtype=None,
                    target_keys=BOOGU_FP8_OPTIMIZATION_TARGET_KEYS,
                    exclude_keys=BOOGU_FP8_OPTIMIZATION_EXCLUDE_KEYS,
                    disable_numpy_memmap=getattr(args, "disable_numpy_memmap", False),
                )
                apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
                if loading_device.type != "cpu":
                    sd = {key: value.to(loading_device) for key, value in sd.items()}
            else:
                sd = load_safetensors(
                    dit_path,
                    device=loading_device,
                    dtype=dit_weight_dtype,
                    disable_numpy_memmap=getattr(args, "disable_numpy_memmap", False),
                )

            info = model.load_state_dict(sd, strict=True, assign=True)
            logger.info(f"Loaded Boogu transformer from safetensors: {info}")
        else:
            if fp8_scaled:
                raise ValueError("Boogu scaled fp8 currently requires a direct .safetensors --dit path.")
            pretrained_kwargs = {"torch_dtype": dit_weight_dtype}
            if os.path.exists(os.path.join(dit_path, "config.json")):
                model = BooguImageTransformer2DModel.from_pretrained(dit_path, **pretrained_kwargs)
            else:
                model = BooguImageTransformer2DModel.from_pretrained(dit_path, subfolder="transformer", **pretrained_kwargs)
            model.to(loading_device, dtype=dit_weight_dtype)

        if attention_backend != "native":
            model.set_attention_backend(attention_backend)
        return model

    def compile_transformer(self, args, transformer):
        model = transformer
        return model_utils.compile_transformer(
            args,
            model,
            [model.noise_refiner, model.context_refiner, model.double_stream_layers, model.single_stream_layers],
            disable_linear=self.blocks_to_swap > 0,
        )

    def scale_shift_latents(self, latents):
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        from musubi_tuner.boogu_image.boogu_utils import (
            boogu_loss_target,
            boogu_raw_velocity_to_musubi_velocity,
            musubi_timestep_to_boogu_time,
            pad_instruction_features,
        )

        model = accelerator.unwrap_model(transformer)
        instruction_features = batch["boogu_instruction_embed"]
        instruction_features, instruction_mask = pad_instruction_features(
            instruction_features,
            device=accelerator.device,
            dtype=network_dtype,
        )
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        boogu_t = musubi_timestep_to_boogu_time(timesteps).to(device=accelerator.device, dtype=torch.float32)
        if boogu_t.dim() == 0:
            boogu_t = boogu_t.unsqueeze(0)
        if boogu_t.shape[0] != noisy_model_input.shape[0]:
            boogu_t = boogu_t.expand(noisy_model_input.shape[0])

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            instruction_features.requires_grad_(True)

        with accelerator.autocast():
            raw_velocity = model(
                hidden_states=noisy_model_input,
                timestep=boogu_t,
                instruction_hidden_states=instruction_features,
                freqs_cis=self._get_freqs_cis(model),
                instruction_attention_mask=instruction_mask,
                ref_image_hidden_states=None,
                return_dict=False,
            )

        model_pred = boogu_raw_velocity_to_musubi_velocity(raw_velocity)
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        target = boogu_loss_target(noise.to(device=accelerator.device, dtype=network_dtype), latents)
        return model_pred, target

    def post_save_checkpoint_hook(self, args, ckpt_file, ckpt_name, accelerator, force_sync_upload=False):
        if not getattr(args, "convert_to_comfy", True):
            return

        try:
            from musubi_tuner.boogu_image.convert_lora_to_comfy import convert_lora_to_comfy

            comfy_ckpt_name = ckpt_name.replace(".safetensors", ".comfy.safetensors")
            comfy_ckpt_file = os.path.join(args.output_dir, comfy_ckpt_name)
            convert_lora_to_comfy(ckpt_file, comfy_ckpt_file, verbose=False)
            accelerator.print(f"Saved ComfyUI-compatible LoRA: {comfy_ckpt_file}")

            if args.huggingface_repo_id is not None:
                from musubi_tuner.utils import huggingface_utils

                huggingface_utils.upload(args, comfy_ckpt_file, "/" + comfy_ckpt_name, force_sync_upload=force_sync_upload)

            if not getattr(args, "save_original_lora", True) and os.path.exists(ckpt_file):
                try:
                    os.remove(ckpt_file)
                    accelerator.print(f"Removed original LoRA checkpoint (--no_save_original_lora): {ckpt_file}")
                except Exception as e:
                    accelerator.print(f"Warning: Failed to remove original checkpoint '{ckpt_file}': {e}")
        except Exception as e:
            accelerator.print(f"Warning: Failed to convert Boogu Image LoRA to ComfyUI format: {e}")


def boogu_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="use dynamic scaled fp8 for the Boogu transformer (requires --fp8_base)",
    )
    parser.add_argument("--text_encoder", type=str, default=None, help="Qwen3-VL text encoder path for cache/sample encoding")
    parser.add_argument("--processor", type=str, default=None, help="Qwen3-VL processor/tokenizer path for Boogu prompt encoding")
    parser.add_argument("--text_encoder_subfolder", type=str, default="auto", help="Qwen3-VL model subfolder for Boogu text encoding")
    parser.add_argument("--max_text_length", type=int, default=1024, help="maximum Qwen3-VL token length for Boogu prompt encoding")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for the Qwen3-VL text encoder during cache/sample encoding")
    return parser


def main():
    parser = setup_parser_common()
    parser = boogu_image_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = BooguImageNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
