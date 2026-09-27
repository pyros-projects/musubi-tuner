"""Cache image latents for Qwen-Image 2.1 training.

Images are encoded by the Qwen-Image 2.1 VAE (RGB gets an opaque alpha channel) and stored
normalized, ``(raw - mean) / std``, as (C, 1, H/16, W/16). Plain text-to-image only.
"""

import logging
from typing import List

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_QWEN_IMAGE21, ItemInfo, save_latent_cache_qwen_image21
from musubi_tuner.qwen_image21.qwen_image21_vae import QwenImage21VAE, load_qwen_image21_vae
from musubi_tuner.utils.model_utils import str_to_dtype
import musubi_tuner.cache_latents as cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(vae: QwenImage21VAE, batch: List[ItemInfo]):
    contents = []
    for item in batch:
        content = item.content
        content = content[0] if isinstance(content, list) else content  # (H, W, C)
        contents.append(torch.from_numpy(content))
    contents = torch.stack(contents, dim=0).permute(0, 3, 1, 2)  # (B, C, H, W)
    contents = contents.float() / 127.5 - 1.0  # [-1, 1]

    latents = vae.encode_pixels_to_latents(contents)  # (B, 64, H/16, W/16)

    for b, item in enumerate(batch):
        target_latent = latents[b].unsqueeze(1)  # (C, 1, H, W)
        print(f"Saving cache for item {item.item_key} at {item.latent_cache_path}, latents shape: {target_latent.shape}")
        save_latent_cache_qwen_image21(item_info=item, latent=target_latent)


def main():
    parser = cache_latents.setup_parser_common()
    parser = cache_latents.hv_setup_parser(parser)  # VAE

    args = parser.parse_args()

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    vae_dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_QWEN_IMAGE21)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    assert args.vae is not None, "VAE checkpoint is required"

    vae = load_qwen_image21_vae(args.vae, device=device, dtype=vae_dtype)

    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch)

    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
