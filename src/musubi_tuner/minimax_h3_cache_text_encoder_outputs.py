import argparse
import logging
import os
import sys
from typing import Optional

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_MINIMAX_H3,
    ARCHITECTURE_MINIMAX_H3_FULL,
    ItemInfo,
    save_text_encoder_output_cache_minimax_h3,
)
from musubi_tuner.minimax_h3.text_encoder import encode_prompts, load_text_encoder, load_tokenizer
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.utils.model_utils import str_to_dtype


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_SAMPLE_PROMPTS_CACHE = "minimax_h3_sample_prompts_cache.pt"
DEFAULT_UNCOND_CACHE = "minimax_h3_uncond_cache.pt"


@torch.no_grad()
def encode_and_save_batch(tokenizer, text_encoder, batch: list[ItemInfo], device, max_length):
    embeds, tags = encode_prompts(tokenizer, text_encoder, [item.caption for item in batch], device, max_length)
    for item, embed, token_tags in zip(batch, embeds, tags):
        save_text_encoder_output_cache_minimax_h3(item, embed, token_tags)


@torch.no_grad()
def encode_prompt_comfy(clip, prompt: str, max_length: int):
    tokens = clip.tokenize(prompt)
    tokens = {key: [row[:max_length] for row in rows] for key, rows in tokens.items()}
    encoded = clip.encode_from_tokens(tokens, return_dict=True)
    embed = encoded["cond"]
    token_tags = encoded.get("minimax_token_tags")
    if embed.ndim != 3 or embed.shape[0] != 1 or token_tags is None:
        raise ValueError("ComfyUI did not return MiniMax H3 conditioning and token tags")
    return embed[0].detach().cpu(), token_tags.detach().cpu()


@torch.no_grad()
def encode_and_save_batch_comfy(clip, batch: list[ItemInfo], cache: dict[str, tuple[torch.Tensor, torch.Tensor]], max_length: int):
    for item in batch:
        if item.caption not in cache:
            if len(cache) >= 32:
                cache.pop(next(iter(cache)))
            cache[item.caption] = encode_prompt_comfy(clip, item.caption, max_length)
        embed, token_tags = cache[item.caption]
        save_text_encoder_output_cache_minimax_h3(item, embed, token_tags)


def resolve_sample_prompts_cache_path(dataset_config: str, datasets=None, filename: str = DEFAULT_SAMPLE_PROMPTS_CACHE) -> str:
    cache_dir = getattr(datasets[0], "cache_directory", None) if datasets else None
    if not cache_dir:
        if not dataset_config:
            raise ValueError("--dataset_config is required to resolve the H3 sample prompt cache")
        declared_datasets = config_utils.load_user_config(dataset_config).get("datasets", [])
        if not declared_datasets or not isinstance(declared_datasets[0], dict):
            raise ValueError("No datasets available to resolve the H3 sample prompt cache")
        cache_dir = declared_datasets[0].get("cache_directory")
    if not cache_dir:
        raise ValueError("First dataset has no cache_directory; set it in the dataset config")
    return os.path.join(os.path.abspath(os.path.expanduser(cache_dir)), filename)


def _text_encoder_cache_metadata(args: argparse.Namespace) -> dict:
    text_encoder = os.path.abspath(os.path.expanduser(args.text_encoder))
    stat = os.stat(text_encoder)
    return {
        "text_encoder": text_encoder,
        "text_encoder_size": stat.st_size,
        "text_encoder_mtime_ns": stat.st_mtime_ns,
        "max_token_length": args.max_token_length,
    }


def load_sample_prompt_cache(cache_path: str, prompts: list[dict], expected_metadata: Optional[dict] = None):
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("Ignoring unreadable H3 sample prompt cache %s: %s", cache_path, exc)
        return None
    if not isinstance(payload, dict) or payload.get("architecture") != ARCHITECTURE_MINIMAX_H3_FULL:
        return None
    if expected_metadata and any(payload.get(key) != value for key, value in expected_metadata.items()):
        return None

    entries = payload.get("prompt_cache")
    if not isinstance(entries, list) or len(entries) != len(prompts):
        return None
    for prompt, entry in zip(prompts, entries):
        if not isinstance(entry, dict) or entry.get("prompt") != prompt.get("prompt", ""):
            return None
        embed = entry.get("h3_text_embed")
        tags = entry.get("h3_token_tags")
        if not isinstance(embed, torch.Tensor) or not isinstance(tags, torch.Tensor):
            return None
        if embed.ndim != 2 or tags.ndim != 1 or embed.shape[0] != tags.shape[0]:
            return None
    return entries


def _precache_sample_prompts(args: argparse.Namespace, datasets, encode_prompt_list) -> None:
    if not args.sample_prompts:
        raise ValueError("--sample_prompts is required with --precache_sample_prompts")
    prompts = load_prompts(args.sample_prompts)
    if not prompts:
        raise ValueError(f"No prompts found in {args.sample_prompts}")
    if any(prompt.get("negative_prompt") is not None for prompt in prompts):
        raise ValueError("MiniMax H3 preview sampling does not support negative prompts")

    cache_path = resolve_sample_prompts_cache_path(args.dataset_config, datasets)
    metadata = _text_encoder_cache_metadata(args)
    if load_sample_prompt_cache(cache_path, prompts, metadata) is not None:
        logger.info("H3 sample prompt cache is current: %s", cache_path)
        return

    prompt_texts = [prompt.get("prompt", "") for prompt in prompts]
    unique_prompt_texts = list(dict.fromkeys(prompt_texts))
    embeds, tags = encode_prompt_list(unique_prompt_texts)
    encoded = dict(zip(unique_prompt_texts, zip(embeds, tags)))
    entries = [
        {
            "prompt": prompt_text,
            "h3_text_embed": encoded[prompt_text][0].detach().cpu().contiguous(),
            "h3_token_tags": encoded[prompt_text][1].detach().cpu().contiguous(),
        }
        for prompt_text in prompt_texts
    ]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "version": 1,
            "architecture": ARCHITECTURE_MINIMAX_H3_FULL,
            **metadata,
            "prompt_cache": entries,
        },
        cache_path,
    )
    logger.info("Saved H3 sample prompt cache: %s", cache_path)


def _precache_uncond(args: argparse.Namespace, datasets, encode_prompt_list) -> None:
    """Cache the empty-prompt (unconditional) embedding for CFG-augmented training."""
    uncond_prompts = [{"prompt": ""}]
    cache_path = resolve_sample_prompts_cache_path(args.dataset_config, datasets, filename=DEFAULT_UNCOND_CACHE)
    metadata = _text_encoder_cache_metadata(args)
    if load_sample_prompt_cache(cache_path, uncond_prompts, metadata) is not None:
        logger.info("H3 uncond cache is current: %s", cache_path)
        return
    embeds, tags = encode_prompt_list([""])
    entries = [
        {
            "prompt": "",
            "h3_text_embed": embeds[0].detach().cpu().contiguous(),
            "h3_token_tags": tags[0].detach().cpu().contiguous(),
        }
    ]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "version": 1,
            "architecture": ARCHITECTURE_MINIMAX_H3_FULL,
            **metadata,
            "prompt_cache": entries,
        },
        cache_path,
    )
    logger.info("Saved H3 uncond cache: %s", cache_path)


def load_comfy_clip(comfyui_path: str, text_encoder_path: str, device: torch.device):
    comfyui_path = os.path.abspath(comfyui_path)
    if not os.path.isfile(os.path.join(comfyui_path, "comfy", "sd.py")):
        raise ValueError(f"ComfyUI source not found at {comfyui_path}")
    sys.path.insert(0, comfyui_path)
    import comfy.sd

    model_options = {"load_device": device}
    if device.type == "cpu":
        model_options["offload_device"] = device
    return comfy.sd.load_clip(
        ckpt_paths=[text_encoder_path],
        clip_type=comfy.sd.CLIPType.MINIMAX,
        model_options=model_options,
    )


def minimax_h3_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="H3 Qwen3-VL safetensors file or model directory")
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="tokenizer directory; defaults to the model bundle/Hugging Face"
    )
    parser.add_argument("--text_encoder_dtype", type=str, default="bfloat16", help="text encoder dtype")
    parser.add_argument("--max_token_length", type=int, default=1024, help="maximum raw prompt token length")
    parser.add_argument("--comfyui_path", type=str, help="use ComfyUI's MiniMax loader, including INT8 text encoders")
    parser.add_argument("--precache_sample_prompts", action="store_true", help="cache H3 training preview prompts")
    parser.add_argument(
        "--precache_uncond",
        action="store_true",
        help="cache the empty-prompt embedding for --cfg_augmented_scale training",
    )
    parser.add_argument("--sample_prompts", type=str, help="sample prompt file to cache")
    parser.add_argument(
        "--cache_sample_prompts_only",
        action="store_true",
        help="cache sample prompts, then exit without scanning dataset text caches",
    )
    parser.add_argument("--disable_numpy_memmap", action="store_true", help="disable memory-mapped checkpoint loading")
    return parser


def main():
    parser = minimax_h3_setup_parser(cache_text_encoder_outputs.setup_parser_common())
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.cache_sample_prompts_only and not args.precache_sample_prompts:
        raise ValueError("--cache_sample_prompts_only requires --precache_sample_prompts")

    datasets = []
    all_files = []
    all_paths = []
    if not args.cache_sample_prompts_only:
        blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
            config_utils.load_user_config(args.dataset_config), args, architecture=ARCHITECTURE_MINIMAX_H3
        )
        datasets = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets
        all_files, all_paths = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    if args.comfyui_path:
        clip = None
        comfy_cache = {}

        def get_clip():
            nonlocal clip
            if clip is None:
                logger.info("Loading MiniMax H3 Qwen3-VL text encoder through ComfyUI")
                clip = load_comfy_clip(args.comfyui_path, args.text_encoder, device)
            return clip

        def encode_prompt_list(prompts):
            current_clip = get_clip()
            for prompt in prompts:
                if prompt not in comfy_cache:
                    comfy_cache[prompt] = encode_prompt_comfy(current_clip, prompt, args.max_token_length)
            values = [comfy_cache[prompt] for prompt in prompts]
            return [value[0] for value in values], [value[1] for value in values]

        def encode(batch):
            encode_and_save_batch_comfy(get_clip(), batch, comfy_cache, args.max_token_length)

    else:
        tokenizer = None
        text_encoder = None
        dtype = str_to_dtype(args.text_encoder_dtype)
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("MiniMax H3 text encoding requires float16, bfloat16, or float32")

        def get_native_encoder():
            nonlocal tokenizer, text_encoder
            if text_encoder is None:
                tokenizer = load_tokenizer(args.tokenizer, args.text_encoder)
                logger.info("Loading MiniMax H3 Qwen3-VL text encoder (layers 0-49)")
                text_encoder = load_text_encoder(
                    args.text_encoder, device=device, dtype=dtype, disable_numpy_memmap=args.disable_numpy_memmap
                )
            return tokenizer, text_encoder

        def encode_prompt_list(prompts):
            current_tokenizer, current_text_encoder = get_native_encoder()
            return encode_prompts(current_tokenizer, current_text_encoder, prompts, device, args.max_token_length)

        def encode(batch):
            current_tokenizer, current_text_encoder = get_native_encoder()
            encode_and_save_batch(current_tokenizer, current_text_encoder, batch, device, args.max_token_length)

    if args.cache_sample_prompts_only:
        _precache_sample_prompts(args, datasets, encode_prompt_list)
        if args.precache_uncond:
            _precache_uncond(args, datasets, encode_prompt_list)
        logger.info("H3 sample prompt cache warmup complete; exiting before training")
        return

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_files,
        all_paths,
        encode,
    )
    cache_text_encoder_outputs.post_process_cache_files(datasets, all_files, all_paths, args.keep_cache)
    if args.precache_sample_prompts:
        _precache_sample_prompts(args, datasets, encode_prompt_list)
    if args.precache_uncond:
        _precache_uncond(args, datasets, encode_prompt_list)
    logger.info("H3 text cache process complete; text encoder memory will be released on process exit")


if __name__ == "__main__":
    main()
