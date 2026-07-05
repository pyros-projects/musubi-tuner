#!/usr/bin/env python3
"""Boogu Image standalone prompt-file inference.

This reuses the training preview sampler without starting a training run, so
Boogu sample prompt TOML files behave the same in both surfaces.
"""

from __future__ import annotations

import argparse
import logging
from types import SimpleNamespace

import torch
from accelerate import Accelerator

from musubi_tuner.boogu_image_train_network import BooguImageNetworkTrainer, boogu_image_setup_parser
from musubi_tuner.utils import model_utils


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Boogu Image generation from training-style sample prompt files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dit", type=str, required=True, help="Boogu transformer checkpoint (.safetensors)")
    parser.add_argument("--vae", type=str, required=True, help="FLUX-compatible VAE checkpoint")
    parser.add_argument(
        "--sample_prompts",
        "--from_file",
        dest="sample_prompts",
        type=str,
        required=True,
        help="Boogu sample prompt TOML/TXT/JSON file, same format as training --sample_prompts",
    )
    parser.add_argument("--output_dir", type=str, default="output", help="Directory where sample/ outputs are saved")
    parser.add_argument("--output_name", type=str, default="boogu_gen", help="Base output filename prefix")

    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Force CPU or CUDA accelerator")
    parser.add_argument("--dit_dtype", type=str, default="bfloat16", help="Transformer weight dtype unless --fp8_scaled is used")
    parser.add_argument("--vae_dtype", type=str, default="bfloat16", help="VAE dtype")
    parser.add_argument("--disable_numpy_memmap", action="store_true", help="Disable safetensors numpy memmap loading")

    parser.add_argument("--attn_mode", type=str, default="sdpa", choices=["sdpa", "torch", "flash", "xformers"])
    parser.add_argument("--sdpa", action="store_true", help="Use SDPA/native attention")
    parser.add_argument("--flash_attn", action="store_true", help="Use FlashAttention")
    parser.add_argument("--xformers", action="store_true", help="Use native attention path for xformers-compatible configs")
    parser.add_argument("--flash3", action="store_true", help="Accepted for config portability; Boogu uses FlashAttention path")
    parser.add_argument("--split_attn", action="store_true", help="Accepted for config portability")

    parser.add_argument("--fp8_base", action="store_true", help="Enable Boogu scaled-fp8 transformer loading")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="Boogu block swap is currently unsupported")
    parser.add_argument("--sample_blocks_to_swap", type=int, default=None, help="Sampling block swap override if supported")
    parser.add_argument("--sample_with_offloading", action="store_true", help="Move transformer to CPU between samples")

    parser.add_argument("--sampling_lora_weight", type=str, nargs="*", default=None, help="Temporary sampling LoRA path(s)")
    parser.add_argument(
        "--sampling_lora_multiplier",
        type=float,
        nargs="*",
        default=None,
        help="Multiplier(s) for temporary sampling LoRA path(s)",
    )
    parser.add_argument("--sample_live_reload_loras", action="store_true", help="Reload sampling LoRA files before sampling")
    parser.add_argument(
        "--network_module",
        type=str,
        default=None,
        help="Network module for sampling LoRA application; Boogu has a default",
    )

    parser.add_argument("--compile", action="store_true", help="Compile the transformer before sampling")
    parser.add_argument("--compile_backend", type=str, default="inductor")
    parser.add_argument("--compile_mode", type=str, default="default")
    parser.add_argument("--compile_dynamic", type=str, default=None)
    parser.add_argument("--compile_fullgraph", action="store_true")
    parser.add_argument("--compile_cache_size_limit", type=int, default=None)

    parser = boogu_image_setup_parser(parser)
    return parser.parse_args(argv)


def _resolve_attn_mode(args: argparse.Namespace) -> str:
    if getattr(args, "flash_attn", False) or getattr(args, "flash3", False):
        return "flash"
    if getattr(args, "sdpa", False):
        return "sdpa"
    if getattr(args, "xformers", False):
        return "xformers"
    return args.attn_mode


def _dtype_from_name(name: str | None, default: torch.dtype) -> torch.dtype:
    if not name:
        return default
    return model_utils.str_to_dtype(name)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.fp8_base and not args.fp8_scaled:
        raise ValueError("Boogu Image fp8 inference supports only scaled fp8: pass --fp8_scaled together with --fp8_base.")

    use_cpu = args.device == "cpu"
    mixed_precision = args.mixed_precision if args.mixed_precision != "no" else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision, cpu=use_cpu)
    device = accelerator.device

    trainer = BooguImageNetworkTrainer()
    trainer.blocks_to_swap = int(args.blocks_to_swap or 0)
    trainer.handle_model_specific_args(args)

    logger.info("Encoding Boogu prompt file: %s", args.sample_prompts)
    sample_parameters = trainer.process_sample_prompts(args, accelerator, args.sample_prompts) or []
    if not sample_parameters:
        logger.warning("No Boogu prompts found in %s", args.sample_prompts)
        return

    attn_mode = _resolve_attn_mode(args)
    dit_weight_dtype = None if args.fp8_scaled else _dtype_from_name(args.dit_dtype, torch.bfloat16)
    logger.info("Loading Boogu transformer: %s", args.dit)
    transformer = trainer.load_transformer(
        accelerator=SimpleNamespace(device=device),
        args=args,
        dit_path=args.dit,
        attn_mode=attn_mode,
        split_attn=bool(getattr(args, "split_attn", False)),
        loading_device=str(device),
        dit_weight_dtype=dit_weight_dtype,
    )
    if args.compile:
        transformer = trainer.compile_transformer(args, transformer)

    vae_dtype = _dtype_from_name(args.vae_dtype, torch.bfloat16)
    vae = trainer.load_vae(args, vae_dtype=vae_dtype, vae_path=args.vae)

    logger.info("Generating %d Boogu sample(s)", len(sample_parameters))
    trainer.sample_images(
        accelerator=accelerator,
        args=args,
        epoch=None,
        steps=0,
        vae=vae,
        transformer=transformer,
        sample_parameters=sample_parameters,
        dit_dtype=trainer.dit_dtype,
        force_sample=True,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
