import argparse
from typing import Optional
import math
import os

import torch
from tqdm import tqdm
from accelerate import Accelerator
from safetensors.torch import load_file, save_file
from safetensors import safe_open

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_Z_IMAGE, ARCHITECTURE_Z_IMAGE_FULL
from musubi_tuner.zimage import zimage_model, zimage_utils, zimage_autoencoder, zimage_config
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    load_prompts,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.utils import model_utils
from musubi_tuner import convert_lora

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ZImageNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_Z_IMAGE

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_Z_IMAGE_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = (
            torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
        )
        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 0.0  # embedded guidance scale. not used for Z-Image model
        self.default_discrete_flow_shift = 3.0  # Z-Image uses flux-shift, so it's better to use it

    def normalize_sampling_lora_weights(
        self, args: argparse.Namespace, weights_sd: dict[str, torch.Tensor], weight_path: str
    ) -> dict[str, torch.Tensor]:
        weights_sd = super().normalize_sampling_lora_weights(args, weights_sd, weight_path)

        packed_qkv_prefixes = sorted(
            {
                key[: -len(".lora_down.weight")]
                for key in weights_sd
                if key.endswith(".lora_down.weight") and "attention_qkv" in key
            }
        )
        if packed_qkv_prefixes:
            logger.info(f"Sampling LoRA {weight_path} uses packed attention_qkv weights; splitting to to_q/to_k/to_v.")
            converted: dict[str, torch.Tensor] = {}
            skip_prefixes = set(packed_qkv_prefixes)
            for key, value in weights_sd.items():
                prefix = key.split(".", 1)[0]
                if prefix in skip_prefixes:
                    continue
                converted[key] = value

            for prefix in packed_qkv_prefixes:
                down_key = f"{prefix}.lora_down.weight"
                up_key = f"{prefix}.lora_up.weight"
                alpha_key = f"{prefix}.alpha"
                down_weight = weights_sd[down_key]
                up_weight = weights_sd[up_key]
                alpha = weights_sd.get(alpha_key)
                up_chunks = torch.tensor_split(up_weight, 3, dim=0)
                for suffix, up_chunk in zip(("to_q", "to_k", "to_v"), up_chunks):
                    new_prefix = prefix.replace("attention_qkv", f"attention_{suffix}")
                    converted[f"{new_prefix}.lora_down.weight"] = down_weight.clone()
                    converted[f"{new_prefix}.lora_up.weight"] = up_chunk.contiguous()
                    if alpha is not None:
                        converted[f"{new_prefix}.alpha"] = alpha.clone() if torch.is_tensor(alpha) else alpha
            weights_sd = converted

        missing_alpha = 0
        for key, value in list(weights_sd.items()):
            if not key.endswith(".lora_down.weight"):
                continue
            prefix = key[: -len(".lora_down.weight")]
            alpha_key = f"{prefix}.alpha"
            if alpha_key in weights_sd:
                continue
            weights_sd[alpha_key] = torch.tensor(float(value.shape[0]), dtype=torch.float32)
            missing_alpha += 1

        if missing_alpha:
            logger.info(
                f"Sampling LoRA {weight_path} is missing alpha for {missing_alpha} modules; defaulting alpha to rank."
            )

        return weights_sd

    def post_save_checkpoint_hook(self, args, ckpt_file, ckpt_name, accelerator, force_sync_upload=False):
        """Convert saved Z-Image LoRA to ComfyUI-compatible diffusers format."""
        if not getattr(args, "convert_to_comfy", True):
            return

        try:
            weights_sd = load_file(ckpt_file)
            converted_sd = convert_lora.convert_to_diffusers("lora_unet_", None, weights_sd)
            comfy_ckpt_name = ckpt_name.replace(".safetensors", ".comfy.safetensors")
            comfy_ckpt_file = os.path.join(args.output_dir, comfy_ckpt_name)

            metadata = None
            with safe_open(ckpt_file, framework="pt") as f:
                metadata = f.metadata()

            save_file(converted_sd, comfy_ckpt_file, metadata=metadata)
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
            accelerator.print(f"Warning: Failed to convert LoRA to ComfyUI format: {e}")

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        device = accelerator.device

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        # Load Qwen3 text encoder
        llm_dtype = torch.bfloat16 if not args.fp8_llm else torch.float8_e4m3fn
        # Use load_qwen3 from zimage_utils (wrapper for Qwen3ForCausalLM)
        tokenizer, text_encoder = zimage_utils.load_qwen3(args.text_encoder, dtype=llm_dtype, device=device, disable_mmap=True)
        text_encoder.eval()

        # Encode prompts
        logger.info("Encoding with Qwen3 text encoder")

        sample_prompts_te_outputs = {}  # (prompt) -> (embed, mask)

        # We process one by one to save memory and reuse logic
        for prompt_dict in prompts:
            if "negative_prompt" not in prompt_dict:
                prompt_dict["negative_prompt"] = ""  # empty negative prompt, this is not used if guidance_scale<=1.0

            for prompt in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                if prompt is None or prompt in sample_prompts_te_outputs:
                    continue

                # encode prompt
                logger.info(f"cache Text Encoder outputs for prompt: {prompt}")

                # get_text_embeds moves inputs to device
                # It returns (embed, mask)
                # embed: [B, Seq, Dim], mask: [B, Seq]
                embed, mask = zimage_utils.get_text_embeds(tokenizer, text_encoder, prompt)

                # Move to CPU to save memory
                embed = embed.cpu()
                mask = mask.cpu()

                sample_prompts_te_outputs[prompt] = (embed, mask)

        del tokenizer, text_encoder
        clean_memory_on_device(device)

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            prompt = prompt_dict.get("prompt", "")
            embed, mask = sample_prompts_te_outputs[prompt]
            prompt_dict_copy["cap_feats"] = embed
            prompt_dict_copy["cap_mask"] = mask

            negative_prompt = prompt_dict.get("negative_prompt", "")
            negative_embed, negative_mask = sample_prompts_te_outputs[negative_prompt]
            prompt_dict_copy["negative_cap_feats"] = negative_embed
            prompt_dict_copy["negative_cap_mask"] = negative_mask

            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)

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
        """architecture dependent inference"""
        model: zimage_model.ZImageTransformer2DModel = accelerator.unwrap_model(transformer)
        device = accelerator.device

        # Get embeddings
        # They are [1, Seq, Dim]
        embed = sample_parameter["cap_feats"].to(device=device, dtype=torch.bfloat16)
        mask = sample_parameter["cap_mask"].to(device=device, dtype=torch.bool)

        if cfg_scale is None:
            cfg_scale = 4.0  # default for Base model
        do_cfg = cfg_scale > 1.0
        if do_cfg:
            negative_embed = sample_parameter["negative_cap_feats"].to(device=device, dtype=torch.bfloat16)
            negative_mask = sample_parameter["negative_cap_mask"].to(device=device, dtype=torch.bool)
        else:
            negative_embed = None
            negative_mask = None

        # Prepare latent variables
        vae_scale = zimage_config.ZIMAGE_VAE_SCALE_FACTOR * 2
        height_latent = 2 * (int(height) // vae_scale)
        width_latent = 2 * (int(width) // vae_scale)
        shape = (1, model.in_channels, height_latent, width_latent)

        latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32).to(device)

        # Trim embeddings
        image_sequence_length = (height_latent // model.all_patch_size[0]) * (width_latent // model.all_patch_size[0])
        embed, _ = zimage_utils.trim_pad_embeds_and_mask(image_sequence_length, embed, mask)
        mask = None  # No attention mask needed after trimming
        if negative_embed is not None:
            negative_embed, _ = zimage_utils.trim_pad_embeds_and_mask(image_sequence_length, negative_embed, negative_mask)
            negative_mask = None  # No attention mask needed after trimming

        # Prepare timesteps
        timesteps, sigmas = zimage_utils.get_timesteps_sigmas(sample_steps, discrete_flow_shift)
        timesteps = timesteps.to(device)
        sigmas = sigmas.to(device)

        # Z-Image inference loop
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000  # Reverse for z-image

            latent_model_input = latents.to(model.dtype)
            latent_model_input = latent_model_input.unsqueeze(2)  # Add frame dimension [B, C, F, H, W]

            with accelerator.autocast(), torch.no_grad():
                model_out = transformer(latent_model_input, timestep, embed, mask)

            if do_cfg:
                with accelerator.autocast(), torch.no_grad():
                    negative_model_out = transformer(latent_model_input, timestep, negative_embed, negative_mask)
                noise_pred = model_out + cfg_scale * (model_out - negative_model_out)
            else:
                noise_pred = model_out

            noise_pred = -noise_pred.squeeze(2)  # Remove frame dimension and invert sign
            latents = zimage_utils.step(noise_pred.to(torch.float32), latents, sigmas, i)

        # Decode
        latents = latents.to(vae.dtype)
        vae.to(device)
        vae.eval()

        logger.info(f"Decoding image from latents: {latents.shape}")
        latents = zimage_utils.shift_scale_latents_for_decode(latents)
        with torch.no_grad():
            pixels = vae.decode(latents)

        pixels = pixels.to(torch.float32).cpu()
        pixels = (pixels / 2 + 0.5).clamp(0, 1)

        vae.to("cpu")
        clean_memory_on_device(device)

        pixels = pixels.unsqueeze(2)  # B C F H W. F=1.
        return pixels

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_path = args.vae
        logger.info(f"Loading VAE model from {vae_path}")
        vae = zimage_autoencoder.load_autoencoder_kl(vae_path, device="cpu", disable_mmap=True)
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
        # zimage_model.load_zimage_model
        model = zimage_model.load_zimage_model(
            device=loading_device,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            dit_weight_dtype=dit_weight_dtype,
            fp8_scaled=args.fp8_scaled,
            disable_numpy_memmap=args.disable_numpy_memmap,
            use_16bit_for_attention=not args.use_32bit_attention,
        )
        return model

    def compile_transformer(self, args, transformer):
        model: zimage_model.ZImageTransformer2DModel = transformer
        # Compile blocks
        return model_utils.compile_transformer(
            args, model, [model.noise_refiner, model.context_refiner, model.layers], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents):
        # Transform VAE latents to Model latents
        # Model Latents = (VAE Latents - shift) * scale
        shift = zimage_config.ZIMAGE_VAE_SHIFT_FACTOR
        scale = zimage_config.ZIMAGE_VAE_SCALING_FACTOR
        latents = (latents - shift) * scale
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
        model: zimage_model.ZImageTransformer2DModel = accelerator.unwrap_model(transformer)
        bsize = latents.shape[0]

        # latents: [B, C, H, W]
        # noisy_model_input: [B, C, H, W]
        image_sequence_length = (latents.shape[2] // model.all_patch_size[0]) * (latents.shape[3] // model.all_patch_size[0])

        # Add frame dimension F=1
        noisy_model_input = noisy_model_input.unsqueeze(2)  # [B, C, 1, H, W]

        # Caption inputs and masks
        llm_embed = batch["llm_embed"]  # list[torch.Tensor]

        txt_seq_lens = [x.shape[0] for x in llm_embed]

        max_len = max(txt_seq_lens)
        # if not split_attn, we need to make attention mask
        if not args.split_attn and bsize > 1:
            padded_len = math.ceil((max_len + image_sequence_length) / zimage_config.SEQ_MULTI_OF) * zimage_config.SEQ_MULTI_OF
            max_len = int(padded_len) - image_sequence_length
            llm_mask = torch.zeros(bsize, max_len, dtype=torch.bool, device=llm_embed[0].device)
            for i, x in enumerate(txt_seq_lens):
                llm_mask[i, :x] = True
        else:
            llm_mask = None  # if split_attn, vl_mask is not used

        llm_embed = [torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in llm_embed]
        llm_embed = torch.stack(llm_embed, dim=0)  # B, L, D

        # print(f"llm_embed shape: {llm_embed.shape}, vl_mask shape: {vl_mask.shape if vl_mask is not None else None}")

        # Timesteps
        t_input = (1000.0 - timesteps) / 1000.0

        # Prepare inputs on device
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        llm_embed = llm_embed.to(device=accelerator.device, dtype=network_dtype)
        llm_mask = llm_mask.to(device=accelerator.device) if llm_mask is not None else None
        t_input = t_input.to(device=accelerator.device, dtype=network_dtype)

        # Enable grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            llm_embed.requires_grad_(True)

        # Call model
        with accelerator.autocast():
            model_pred = transformer(x=noisy_model_input, t=t_input, cap_feats=llm_embed, cap_mask=llm_mask)

        # model_pred: [B, C, F, H, W]
        model_pred = model_pred.squeeze(2)  # [B, C, H, W]

        # Target: Opposite of usual Flow matching
        target = latents - noise

        return model_pred, target


def zimage_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Z-Image specific parser setup"""
    # parser.add_argument("--dit_dtype", type=str, default=None, help="data type for DiT, default is bfloat16")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT")
    parser.add_argument("--text_encoder", type=str, default=None, help="Qwen3 text encoder checkpoint path")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument(
        "--use_32bit_attention",
        action="store_true",
        help="use 32-bit precision for attention computations in DiT model even when using mixed precision (original behavior)",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = zimage_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if args.vae_dtype is not None:
        logger.warning("vae_dtype is not used in Z-Image architecture (always float32)")

    trainer = ZImageNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
