"""LoRA training for Krea 2 (K2). Phase 4: minimal trainable loop.

Implements the architecture-specific hooks of the shared NetworkTrainer for K2:
build/load the single-stream MMDiT, reuse the Qwen-Image VAE, and run flow-matching
in K2's token space (replicating sampling.prepare batched, with varlen text padding).

Wired: bf16 + gradient checkpointing, sample generation during training (text-to-image,
optional CFG), dynamic scaled fp8 for the DiT (--fp8_base --fp8_scaled), block swap
(--blocks_to_swap, CPU offloading of the main blocks), and torch.compile of the main
blocks (--compile).
"""

import argparse
import gc
import os
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from einops import rearrange, repeat

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_KREA2, ARCHITECTURE_KREA2_FULL
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    clean_memory_on_device,
    load_prompts,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.krea2 import krea2_utils
from musubi_tuner.krea2 import krea2_sampling
from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.utils import model_utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Krea2NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_frame_stride = 1  # single image

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_KREA2

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_KREA2_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0  # K2 t2i, not used at train time
        # self.blocks_to_swap is set by the base trainer (handle_model_specific_args runs first).
        # K2 fp8 supports only the scaled (dynamic) path; plain --fp8_base alone would cast the
        # whole DiT (incl. norms) to fp8, which breaks. Require --fp8_scaled with --fp8_base.
        if args.fp8_base and not args.fp8_scaled:
            raise ValueError("Krea 2 fp8 supports only scaled fp8: pass --fp8_scaled together with --fp8_base.")

    def _default_sampling_lora_network_module_name(self) -> Optional[str]:
        return "musubi_tuner.networks.lora_krea2"

    def post_save_checkpoint_hook(self, args, ckpt_file, ckpt_name, accelerator, force_sync_upload=False):
        """Convert saved Krea2 LoRA checkpoints to native ComfyUI format."""
        if not getattr(args, "convert_to_comfy", True):
            return

        try:
            from musubi_tuner.krea2.convert_lora_to_comfy import convert_lora_to_comfy

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
            accelerator.print(f"Warning: Failed to convert Krea2 LoRA to ComfyUI format: {e}")

    def normalize_sampling_lora_weights(
        self, args: argparse.Namespace, weights_sd: dict[str, torch.Tensor], weight_path: str
    ) -> dict[str, torch.Tensor]:
        native_prefixes = sorted(
            {
                key[: -len(".lora_down.weight")]
                for key in weights_sd
                if key.startswith("diffusion_model.") and key.endswith(".lora_down.weight")
            }
        )
        if native_prefixes:
            logger.info(
                f"Sampling LoRA {weight_path} uses native Krea2 diffusion_model paths; converting to lora_unet_* sampling format."
            )
            converted: dict[str, torch.Tensor] = {}
            native_prefix_set = set(native_prefixes)
            for key, value in weights_sd.items():
                if any(key == prefix or key.startswith(f"{prefix}.") for prefix in native_prefix_set):
                    continue
                converted[key] = value

            for prefix in native_prefixes:
                module_path = prefix[len("diffusion_model.") :]
                normalized_prefix = f"lora_unet_{module_path.replace('.', '_')}"
                for suffix in ("lora_down.weight", "lora_up.weight", "alpha"):
                    source_key = f"{prefix}.{suffix}"
                    if source_key not in weights_sd:
                        continue
                    value = weights_sd[source_key]
                    converted[f"{normalized_prefix}.{suffix}"] = value.clone() if torch.is_tensor(value) else value
            weights_sd = converted

        weights_sd = super().normalize_sampling_lora_weights(args, weights_sd, weight_path)

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

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        """Encode the sample prompts with Qwen3-VL up front, cache the embeds, free the encoder.

        Kept deliberately simple (text-to-image only): for each prompt and its optional
        negative prompt we store the varlen selected-layer hidden stack (valid tokens only),
        matching the training cache format, then drop the 4B encoder before training resumes.
        """
        device = accelerator.device

        assert args.text_encoder is not None, "--text_encoder is required for sample generation during training"
        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        encoder = krea2_utils.load_krea2_text_encoder(args.text_encoder, dtype=torch.bfloat16, device=device)

        logger.info("Encoding sample prompts with Qwen3-VL")
        te_outputs = {}  # prompt str -> (valid_len, num_layers, hidden) on cpu
        with torch.no_grad():
            for prompt_dict in prompts:
                for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", None)]:
                    if p is None or p in te_outputs:
                        continue
                    hiddens, mask = krea2_utils.get_krea2_prompt_embeds(encoder, [p])  # (1, seq, L, D), (1, seq)
                    embed = hiddens[0][mask[0]]  # gather valid tokens -> (valid_len, L, D), drops padding
                    te_outputs[p] = embed.to("cpu")

        del encoder
        gc.collect()
        clean_memory_on_device(device)

        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            prompt_dict_copy["krea2_vl_embed"] = te_outputs[prompt_dict.get("prompt", "")]
            negative_prompt = prompt_dict.get("negative_prompt", None)
            if negative_prompt is not None:
                prompt_dict_copy["negative_krea2_vl_embed"] = te_outputs[negative_prompt]
            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(device)
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
        """Architecture-dependent inference: K2 flow-matching Euler sampler with optional CFG.

        Replicates krea2_sampling.sample's core loop but with embeds taken from the cached
        sample parameters (no encoder) and the trainer's DiT + Qwen-Image VAE. CFG (standard
        uncond + scale*(cond-uncond)) is enabled when a negative prompt is present and cfg_scale > 1
        (musubi convention).
        Resolution-aware mu time-shift uses the K2 raw defaults (y1=0.5, y2=1.15); the distilled
        (turbo) fixed-mu schedule is not wired here.
        """
        model = transformer  # SingleStreamDiT
        device = accelerator.device
        patch = model.config.patch
        compression = qwen_image_utils.VAE_SCALE_FACTOR  # Qwen-Image VAE: 8x

        # Standard CFG (uncond + scale*(cond-uncond)), enabled when cfg_scale > 1 — matches the
        # rest of musubi. The official Krea 2 "guidance" value maps as cfg_scale = guidance + 1
        # (official default guidance 4.5 -> cfg_scale 5.5).
        cfg = cfg_scale if cfg_scale is not None else 5.5
        do_cfg = do_classifier_free_guidance and cfg > 1.0

        # The latent grid is patchified in `patch`-sized blocks, so spatial dims must be
        # multiples of compression * patch. The base flow already rounds to 8; align to 16.
        align = compression * patch
        width = krea2_sampling.roundup(width, align, "width")
        height = krea2_sampling.roundup(height, align, "height")
        lat_h, lat_w = height // compression, width // compression

        def build_branch(embed):
            embed = embed.to(device=device, dtype=torch.bfloat16).unsqueeze(0)  # (1, seq, L, D)
            txtmask = torch.ones(1, embed.shape[1], device=device, dtype=torch.bool)
            return embed, txtmask

        txt, txtmask = build_branch(sample_parameter["krea2_vl_embed"])
        if do_cfg:
            untxt, untxtmask = build_branch(sample_parameter["negative_krea2_vl_embed"])

        # Seeded gaussian latent noise (generator already seeded by the base sampler).
        noise = torch.randn(1, model.config.channels, lat_h, lat_w, device=device, dtype=torch.bfloat16, generator=generator)

        img, pos, mask = krea2_sampling.prepare(noise, txt.shape[1], patch, txtmask)
        if do_cfg:
            _, unpos, unmask = krea2_sampling.prepare(noise, untxt.shape[1], patch, untxtmask)

        # mu interpolation endpoints (krea2 sample defaults minres=256, maxres=1280).
        x1 = (256 // align) ** 2
        x2 = (1280 // align) ** 2
        sample_mu = sample_parameter.get("mu", None)
        if sample_mu is not None:
            logger.info(f"Krea 2 sample uses fixed mu={sample_mu}")
        ts = krea2_sampling.timesteps(img.shape[1], sample_steps, x1, x2, y1=0.5, y2=1.15, mu=sample_mu)

        for tcurr, tprev in tqdm(zip(ts[:-1], ts[1:]), total=len(ts) - 1, desc="Denoising steps"):
            t = torch.full((1,), tcurr, dtype=img.dtype, device=device)
            cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
            if do_cfg:
                uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
                v = uncond + cfg * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v

        # Unpatchify token sequence back to a latent (1, C, 1, H, W) for the VAE.
        latent = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=lat_h // patch, w=lat_w // patch)
        latent = latent.unsqueeze(2)  # (1, C, 1, H, W)

        vae.to(device)
        vae.eval()
        with torch.no_grad():
            pixels = vae.decode_to_pixels(latent.to(vae.dtype))  # (1, C, H, W) in [0, 1]
        vae.to("cpu")
        clean_memory_on_device(device)

        pixels = pixels.unsqueeze(2).to(torch.float32).cpu()  # (1, C, 1, H, W) for the grid saver
        return pixels

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        logger.info(f"Loading VAE model from {args.vae}")
        vae = qwen_image_utils.load_vae(args.vae, input_channels=3, device="cpu", disable_mmap=True)
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
        # For fp8_scaled, dit_weight_dtype is None (the base trainer skips the post-load cast);
        # the fp8 path ignores dtype and keeps non-target weights in their checkpoint dtype.
        dtype = dit_weight_dtype if dit_weight_dtype is not None else torch.bfloat16
        model = krea2_utils.load_krea2_dit(
            dit_path,
            device=loading_device,
            dtype=dtype,
            fp8_scaled=args.fp8_scaled,
            loading_device=loading_device,
            attn_mode=attn_mode,
            split_attn=split_attn,
        )
        return model

    def compile_transformer(self, args, transformer):
        model = transformer  # SingleStreamDiT
        # Compile the per-block SingleStreamBlocks (the heavy, repeated compute). The forward
        # already pads the combined sequence to a multiple of 256 to keep kernel shapes stable.
        # When block swap is on, exclude the swap blocks' Linears from compile (cf. zimage/qwen_image).
        return model_utils.compile_transformer(args, model, [model.blocks], disable_linear=self.blocks_to_swap > 0)

    def scale_shift_latents(self, latents):
        # K2 latents are already normalized by the Qwen-Image VAE caching ((raw-mean)/std).
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
        model = transformer  # SingleStreamDiT
        device = accelerator.device
        patch = model.config.patch

        latents = batch["latents"]  # (B, C, 1, H, W)
        bsize = latents.shape[0]
        assert latents.shape[2] == 1, f"K2 expects single-frame latents (B,C,1,H,W), got {latents.shape}"

        # --- image tokens / pos / mask (replicates krea2 sampling.prepare) ---
        nmi = noisy_model_input.squeeze(2)  # (B, C, H, W)
        _, _, lat_h, lat_w = nmi.shape
        h_, w_ = lat_h // patch, lat_w // patch

        img_tokens = rearrange(nmi, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

        imgids = torch.zeros((h_, w_, 3), device=device)
        imgids[..., 1] = torch.arange(h_, device=device)[:, None]
        imgids[..., 2] = torch.arange(w_, device=device)[None, :]
        imgpos = repeat(imgids, "h w three -> b (h w) three", b=bsize, three=3)
        imgmask = torch.ones(bsize, h_ * w_, device=device, dtype=torch.bool)

        # --- text tokens / pos / mask (varlen -> padded batch) ---
        vl_embed = batch["krea2_vl_embed"]  # list of (valid_len, num_layers, hidden)
        txt_seq_lens = [x.shape[0] for x in vl_embed]
        max_len = max(txt_seq_lens)
        # pad along the sequence axis (dim 0): F.pad pads last dim first, so (0,0)x2 then (0, pad)
        vl_embed = [F.pad(x, (0, 0, 0, 0, 0, max_len - x.shape[0])) for x in vl_embed]
        context = torch.stack(vl_embed, dim=0).to(device=device, dtype=network_dtype)  # (B, max_len, L, D)

        txtmask = torch.zeros(bsize, max_len, device=device, dtype=torch.bool)
        for i, n in enumerate(txt_seq_lens):
            txtmask[i, :n] = True
        txtpos = torch.zeros(bsize, max_len, 3, device=device)

        # --- combine (image-first: valid tokens form a contiguous prefix per sample) ---
        mask = torch.cat((imgmask, txtmask), dim=1)
        pos = torch.cat((imgpos, txtpos), dim=1)

        img_tokens = img_tokens.to(device=device, dtype=network_dtype)
        t = (timesteps / 1000.0).to(device=device)

        if args.gradient_checkpointing:
            img_tokens.requires_grad_(True)
            context.requires_grad_(True)

        with accelerator.autocast():
            model_pred = model(img=img_tokens, context=context, t=t, pos=pos, mask=mask)  # (B, h*w, c*ph*pw)

        # unpatchify to latent space (B, C, 1, H, W)
        model_pred = rearrange(model_pred, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=h_, w=w_)
        model_pred = model_pred.unsqueeze(2)  # (B, C, 1, H, W)

        # flow matching target (velocity): noise - data
        latents = latents.to(device=device, dtype=network_dtype)
        target = noise - latents
        return model_pred, target

    # endregion model specific


def krea2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="use dynamic scaled fp8 for the DiT (requires --fp8_base). Quantizes per-block Linears at load time.",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="Qwen3-VL-4B text encoder safetensors path (only needed for sample generation during training)",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Krea2NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
