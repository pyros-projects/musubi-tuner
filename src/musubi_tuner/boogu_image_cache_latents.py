"""Cache Boogu Image Base latents for LoRA training."""

from __future__ import annotations

import logging
from typing import List

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_BOOGU_IMAGE, ItemInfo, save_latent_cache_boogu_image
from musubi_tuner.boogu_image.boogu_utils import load_boogu_autoencoder_kl
import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents_boogu_image(batch: List[ItemInfo]) -> torch.Tensor:
    contents = []
    for item in batch:
        content = item.content
        content = content[0] if isinstance(content, list) else content
        contents.append(torch.from_numpy(content[..., :3]))
    tensor = torch.stack(contents, dim=0)
    tensor = tensor.permute(0, 3, 1, 2).to(dtype=torch.float32)
    return tensor / 127.5 - 1.0


def normalize_boogu_vae_latents(latents: torch.Tensor, scaling_factor: float, shift_factor: float | None) -> torch.Tensor:
    shift = 0.0 if shift_factor is None else float(shift_factor)
    return (latents - shift) * float(scaling_factor)


def encode_and_save_batch(vae, batch: List[ItemInfo]):
    contents = preprocess_contents_boogu_image(batch)
    device = getattr(vae, "device", torch.device("cpu"))
    dtype = getattr(vae, "dtype", torch.float32)

    with torch.no_grad():
        encoded = vae.encode(contents.to(device=device, dtype=dtype)).latent_dist.sample()
        config = getattr(vae, "config", {})
        scaling_factor = config.get("scaling_factor", 0.3611)
        shift_factor = config.get("shift_factor", 0.1159)
        latents = normalize_boogu_vae_latents(encoded, scaling_factor, shift_factor)

    for batch_index, item in enumerate(batch):
        target_latent = latents[batch_index].detach().cpu()
        print(f"Saving cache for item {item.item_key} at {item.latent_cache_path}, latents shape: {target_latent.shape}")
        save_latent_cache_boogu_image(item_info=item, latent=target_latent)


def load_boogu_vae(path: str, dtype: torch.dtype, device: torch.device):
    vae = load_boogu_autoencoder_kl(path, dtype)
    vae.to(device=device, dtype=dtype)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def main():
    parser = cache_latents.setup_parser_common()
    parser = cache_latents.hv_setup_parser(parser)
    args = parser.parse_args()

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    vae_dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_BOOGU_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16)
        return

    assert args.vae is not None, "Boogu-compatible VAE path is required"
    logger.info(f"Loading Boogu VAE from {args.vae}")
    vae = load_boogu_vae(args.vae, vae_dtype, device)

    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch)

    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
