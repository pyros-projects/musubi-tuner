"""LoRA training entry point for Boogu Image Base."""

from __future__ import annotations

import argparse
import logging
import os
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

    def process_sample_prompts(self, args: argparse.Namespace, accelerator: Accelerator, sample_prompts: str):
        assert args.text_encoder is not None, "--text_encoder is required for Boogu sample generation during training"

        from musubi_tuner.boogu_image_cache_text_encoder_outputs import build_boogu_t2i_messages, load_boogu_text_encoder

        device = accelerator.device
        prompts = load_prompts(sample_prompts)
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
        text_features: dict[str, torch.Tensor] = {}
        logger.info(f"Encoding Boogu sample prompts with Qwen3-VL: {sample_prompts}")
        with torch.no_grad():
            for prompt_dict in prompts:
                if "negative_prompt" not in prompt_dict:
                    prompt_dict["negative_prompt"] = ""
                for prompt in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                    if prompt is None or prompt in text_features:
                        continue
                    inputs = processor.apply_chat_template(
                        [build_boogu_t2i_messages(prompt)],
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        add_generation_prompt=False,
                        truncation=True,
                        max_length=max_length,
                    )
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    output = text_encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                    valid_len = int(inputs["attention_mask"][0].to(dtype=torch.bool).sum().item())
                    text_features[prompt] = output.last_hidden_state[0, :valid_len].to(torch.bfloat16).cpu()

        del text_encoder
        clean_memory_on_device(device)

        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            prompt_dict_copy["boogu_instruction_embed"] = text_features[prompt_dict.get("prompt", "")]
            negative_prompt = prompt_dict.get("negative_prompt", "")
            prompt_dict_copy["negative_boogu_instruction_embed"] = text_features[negative_prompt]
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

        freqs_cis = self._get_freqs_cis(model)
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
                    ref_image_hidden_states=None,
                    return_dict=False,
                )
                if do_cfg:
                    v_uncond = model(
                        hidden_states=latent_model_input,
                        timestep=boogu_t,
                        instruction_hidden_states=uncond_feats,
                        freqs_cis=freqs_cis,
                        instruction_attention_mask=uncond_mask,
                        ref_image_hidden_states=None,
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
