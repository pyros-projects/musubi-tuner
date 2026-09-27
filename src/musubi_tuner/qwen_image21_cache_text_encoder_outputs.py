"""Cache text encoder (Qwen3-VL-8B) outputs for Qwen-Image 2.1 training.

Each caption is stored as its (tokens, 4096) last-layer states, pre-final-norm, with the system
turn dropped: exactly the context the DiT reads at inference.
"""

import argparse
import logging

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_QWEN_IMAGE21,
    ItemInfo,
    save_text_encoder_output_cache_qwen_image21,
)
from musubi_tuner.qwen_image21 import qwen_image21_text_encoder

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(tokenizer, text_encoder, batch: list[ItemInfo]):
    prompts = [item.caption for item in batch]
    for i, item in enumerate(batch):
        print(f"Item {i}: {item.item_key}, prompt: {item.caption}")

    embeds = qwen_image21_text_encoder.encode_prompts(tokenizer, text_encoder, prompts)
    for item, embed in zip(batch, embeds):
        save_text_encoder_output_cache_qwen_image21(item, embed)


def qwen_image21_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="Qwen3-VL-8B text encoder safetensors path (Comfy bf16 or INT8 ConvRot)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=qwen_image21_text_encoder.QWEN3_VL_TOKENIZER_REPO,
        help="Qwen3-VL tokenizer repo id or local directory",
    )
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = qwen_image21_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_QWEN_IMAGE21)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    tokenizer = qwen_image21_text_encoder.load_tokenizer(args.tokenizer)
    text_encoder = qwen_image21_text_encoder.load_qwen_image21_text_encoder(args.text_encoder, device=device)

    logger.info("Encoding with Qwen3-VL-8B")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        encode_and_save_batch(tokenizer, text_encoder, batch)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )

    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


if __name__ == "__main__":
    main()
