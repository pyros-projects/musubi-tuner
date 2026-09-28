"""LoRA training for Qwen-Image 2.1 (text-to-image), directly on the INT8 ConvRot DiT.

The Comfy ``qwen_image_2.1_int8_convrot`` file is loaded as is: its quantized Linears stay
frozen INT8 and run through comfy-kitchen kernels (input gradients via chunked dequantization,
see ``modules/int8_optimization_utils``), LoRA deltas train in the usual dtype on top. A bf16
Comfy file with fused ``img_mlp.gate_up`` works too.

Training snapshots follow the official pipeline: Euler, resolution-dependent mu with the
``shift_terminal`` stretch, no CFG unless a negative prompt is given with ``cfg_scale`` > 1.
"""

import argparse
import gc
import os
import re
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import load_file
from tqdm import tqdm

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_QWEN_IMAGE21, ARCHITECTURE_QWEN_IMAGE21_FULL
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    clean_memory_on_device,
    load_prompts,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.krea2 import krea2_sampling
from musubi_tuner.networks import lora_qwen_image21
from musubi_tuner.qwen_image21 import qwen_image21_text_encoder, qwen_image21_utils
from musubi_tuner.qwen_image21.qwen_image21_vae import SPATIAL_COMPRESSION, load_qwen_image21_vae

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_overlay_spec(spec: str) -> tuple[str, float]:
    """``PATH`` or ``PATH:STRENGTH`` (H3's --train_lora_overlay convention)."""
    match = re.fullmatch(r"(.+):([0-9]*\.?[0-9]+)", spec)
    return (match.group(1), float(match.group(2))) if match else (spec, 1.0)


def attach_train_lora_overlay(model: torch.nn.Module, path: str, strength: float, device) -> "lora_qwen_image21.lora.LoRANetwork":
    """Hook a frozen training adapter (SimpleTuner's assistant, Fizgig's adapter, ...) onto the DiT.

    Unmerged: every adapted Linear computes W x + strength * B A x, as SimpleTuner and Fizgig
    apply theirs. The trained LoRA wraps these modules afterwards and learns on top. The
    adapter is not part of the trained network, so it never reaches the optimizer or a save.
    """
    weights = qwen_image21_utils.convert_lora_to_musubi(load_file(path), qwen_image21_utils.peft_lora_scale(path))
    network = lora_qwen_image21.create_arch_network_from_weights(strength, weights, unet=model, for_inference=True)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    info = network.load_state_dict(weights, strict=False)
    if info.missing_keys or info.unexpected_keys:
        raise ValueError(f"Training adapter {path} does not match the DiT: {info}")
    network.to(device=device, dtype=model.dtype)
    network.requires_grad_(False)
    return network


def set_train_lora_overlay_enabled(network, enabled: bool) -> None:
    for lora in network.unet_loras:
        lora.enabled = enabled


class QwenImage21NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_frame_stride = 1  # single image
        self.sample_raw_sigmas = None  # --sample_raw_sigmas
        self.train_lora_overlay = None  # --train_lora_overlay network, active in training forwards only

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_QWEN_IMAGE21

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_QWEN_IMAGE21_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0  # the official pipeline samples without guidance
        if args.fp8_base:
            raise ValueError("Qwen-Image 2.1 trains on the INT8 ConvRot (or bf16) checkpoint as is; drop --fp8_base.")
        if args.blocks_to_swap:
            raise ValueError("Qwen-Image 2.1 does not support --blocks_to_swap (the INT8 DiT is about 7.3 GB).")
        if args.compile:
            raise ValueError("Qwen-Image 2.1 does not support --compile.")
        if args.sample_raw_sigmas:
            self.sample_raw_sigmas = [float(s) for s in args.sample_raw_sigmas.split(",")]
        if args.train_lora_overlay:
            path, strength = parse_overlay_spec(args.train_lora_overlay)
            if not os.path.isfile(path):
                raise ValueError(f"--train_lora_overlay file not found: {path}")
            if strength <= 0:
                raise ValueError(f"--train_lora_overlay strength must be positive: {args.train_lora_overlay}")

    def get_checkpoint_metadata(self, args: argparse.Namespace) -> dict[str, str]:
        if not args.train_lora_overlay:
            return {}
        path, strength = parse_overlay_spec(args.train_lora_overlay)
        return {"ss_qwen21_train_lora_overlay": os.path.basename(path), "ss_qwen21_train_lora_overlay_strength": f"{strength:g}"}

    def sample_images(self, *args, **kwargs):
        # previews show what the LoRA does on the plain base, i.e. without the training adapter
        if self.train_lora_overlay is None:
            return super().sample_images(*args, **kwargs)
        set_train_lora_overlay_enabled(self.train_lora_overlay, False)
        try:
            return super().sample_images(*args, **kwargs)
        finally:
            set_train_lora_overlay_enabled(self.train_lora_overlay, True)

    def _default_sampling_lora_network_module_name(self) -> Optional[str]:
        return "musubi_tuner.networks.lora_qwen_image21"

    def normalize_sampling_lora_weights(
        self, args: argparse.Namespace, weights_sd: dict[str, torch.Tensor], weight_path: str
    ) -> dict[str, torch.Tensor]:
        # PEFT/diffusers turbo LoRAs address the unfused gate_layer/proj pair; fuse them onto gate_up
        return qwen_image21_utils.convert_lora_to_musubi(weights_sd, qwen_image21_utils.peft_lora_scale(weight_path))

    def _apply_sampling_lora_network(self, network, weights_sd: dict[str, torch.Tensor], device: torch.device):
        """Attach every sampling LoRA as an unmerged runtime overlay, never merge it: INT8 payloads cannot take
        a delta, and merging a turbo LoRA into bf16 weights loses a large part of it (Viggle's own warning)."""
        attached = 0
        for lora, sd_for_lora in self._iter_sampling_lora_module_weights(network, weights_sd):
            org_module = lora.org_module_ref[0]
            self._attach_sampling_lora_runtime_overlay(
                org_module,
                sd_for_lora["lora_down.weight"],
                sd_for_lora["lora_up.weight"],
                lora.multiplier * float(lora.scale),
                device=org_module.weight.device,
                compute_dtype=torch.bfloat16,
            )
            attached += 1
        return 0, attached, {}

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        """Encode the sample prompts (and negatives) with Qwen3-VL-8B up front, then free the encoder."""
        device = accelerator.device

        assert args.text_encoder is not None, "--text_encoder is required for sample generation during training"
        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        tokenizer = qwen_image21_text_encoder.load_tokenizer(args.tokenizer)
        text_encoder = qwen_image21_text_encoder.load_qwen_image21_text_encoder(args.text_encoder, device=device)

        logger.info("Encoding sample prompts with Qwen3-VL-8B")
        te_outputs = {}  # prompt str -> (tokens, 4096) on cpu
        for prompt_dict in prompts:
            for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", None)]:
                if p is None or p in te_outputs:
                    continue
                te_outputs[p] = qwen_image21_text_encoder.encode_prompts(tokenizer, text_encoder, [p])[0].cpu()

        del text_encoder
        gc.collect()
        clean_memory_on_device(device)

        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            prompt_dict_copy["qwen21_vl_embed"] = te_outputs[prompt_dict.get("prompt", "")]
            negative_prompt = prompt_dict.get("negative_prompt", None)
            if negative_prompt is not None:
                prompt_dict_copy["negative_qwen21_vl_embed"] = te_outputs[negative_prompt]
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
        """Euler flow sampling with the official schedule; true CFG only with a negative prompt and cfg_scale > 1.
        A ``mu`` key in the prompt file pins the time shift instead of deriving it from the resolution.
        ``--sample_raw_sigmas`` (a turbo LoRA's raw nodes) replaces ``sample_steps``."""
        model = transformer
        device = accelerator.device
        cfg = cfg_scale if cfg_scale is not None else 1.0
        do_cfg = do_classifier_free_guidance and cfg > 1.0

        width = krea2_sampling.roundup(width, SPATIAL_COMPRESSION, "width")
        height = krea2_sampling.roundup(height, SPATIAL_COMPRESSION, "height")
        sample_mu = sample_parameter.get("mu", None)
        if sample_mu is not None:
            logger.info(f"Qwen-Image 2.1 sample uses fixed mu={sample_mu}")

        latents = qwen_image21_utils.denoise(
            model,
            sample_parameter["qwen21_vl_embed"],
            width,
            height,
            sample_steps,
            generator,
            negative_context=sample_parameter["negative_qwen21_vl_embed"] if do_cfg else None,
            cfg_scale=cfg,
            mu=sample_mu,
            raw_sigmas=self.sample_raw_sigmas,
            progress=tqdm,
        )

        with krea2_sampling.transformer_decode_offload(
            model,
            device,
            enabled=bool(getattr(args, "sample_with_offloading", False)),
            restore_after=False,
            clean_fn=clean_memory_on_device,
        ):
            vae.to(device)
            try:
                pixels = vae.decode_to_pixels(latents)  # (1, 3, H, W) in [0, 1]
            finally:
                vae.to("cpu")
        clean_memory_on_device(device)

        return pixels.unsqueeze(2).to(torch.float32).cpu()  # (1, C, 1, H, W) for the grid saver

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        logger.info(f"Loading VAE model from {vae_path}")
        return load_qwen_image21_vae(vae_path, device="cpu", dtype=vae_dtype)

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
        if attn_mode != "torch" or split_attn:
            raise ValueError("Qwen-Image 2.1 uses PyTorch SDPA attention: pass --sdpa (without --split_attn).")
        model = qwen_image21_utils.load_qwen_image21_dit(dit_path, device=loading_device, dtype=torch.bfloat16)
        if args.train_lora_overlay:
            # attached before the trained LoRA, which then wraps these modules and learns on top
            path, strength = parse_overlay_spec(args.train_lora_overlay)
            self.train_lora_overlay = attach_train_lora_overlay(model, path, strength, loading_device)
            logger.info(
                f"Training LoRA overlay {path} at strength {strength:g}: {len(self.train_lora_overlay.unet_loras)} modules, "
                "active in training forwards, off for previews, never saved"
            )
        return model

    def scale_shift_latents(self, latents):
        # latents are cached normalized ((raw - mean) / std)
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
        model = transformer
        device = accelerator.device
        dtype = model.dtype  # activations stay in the base dtype; LoRA weights may be fp32

        assert latents.shape[2] == 1, f"Qwen-Image 2.1 expects single-frame latents (B, C, 1, H, W), got {latents.shape}"
        x = noisy_model_input.squeeze(2).to(device=device, dtype=dtype)

        # varlen text -> right-padded batch; the DiT masks the padding for image queries
        embeds = batch["qwen21_vl_embed"]
        txt_lens = [e.shape[0] for e in embeds]
        max_len = max(txt_lens)
        context = torch.stack([F.pad(e, (0, 0, 0, max_len - e.shape[0])) for e in embeds]).to(device=device, dtype=dtype)

        with accelerator.autocast():
            model_pred = model(x, (timesteps / 1000.0).to(device), context, txt_lens=txt_lens)

        # flow matching target (velocity): noise - data
        target = noise - latents.to(device=device, dtype=network_dtype)
        return model_pred.unsqueeze(2), target

    # endregion model specific


def qwen_image21_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="Qwen3-VL-8B text encoder safetensors path (only needed for sample generation during training)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=qwen_image21_text_encoder.QWEN3_VL_TOKENIZER_REPO,
        help="Qwen3-VL tokenizer repo id or local directory",
    )
    parser.add_argument(
        "--sample_raw_sigmas",
        type=str,
        default=None,
        help="comma-separated raw sigma nodes for sample generation, e.g. a turbo LoRA's schedule "
        "(Viggle v0.2.1: 1.0,0.9375,0.875,0.75,0.5,0.25). They get the resolution shift but no terminal "
        "stretch, and replace the prompt file's sample_steps.",
    )
    parser.add_argument(
        "--train_lora_overlay",
        type=str,
        default=None,
        help="frozen training adapter (de-distillation assistant) applied during TRAINING forwards only, PATH[:STRENGTH], "
        "e.g. Fizgig's or SimpleTuner's Qwen-Image 2.1 training adapter. Unmerged, off for previews, never saved, "
        "so the trained LoRA is used on the plain base",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = qwen_image21_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"
    args.fp8_scaled = False  # the base trainer reads it; INT8 ConvRot needs no fp8 path
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = QwenImage21NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
