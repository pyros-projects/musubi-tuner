"""Cache Boogu Image Base Qwen3-VL instruction features."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import copy
import logging
import os

import torch
from accelerate import init_empty_weights

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_BOOGU_IMAGE, ItemInfo, save_text_encoder_output_cache_boogu_image
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.krea2.krea2_encoder import QWEN3_VL_4B_INSTRUCT_CONFIG
from musubi_tuner.utils.safetensors_utils import load_split_weights

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SYSTEM_PROMPT_T2I = (
    "You are a helpful assistant that generates high-quality images based on user "
    "instructions. The instructions are as follows."
)
SYSTEM_PROMPT_DROP = (
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the original input where appropriate."
)
QWEN3_VL_8B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"


@dataclass(frozen=True)
class BooguTextEncoderLoadPlan:
    kind: str
    model_path: str
    processor_path: str
    model_subfolder: str | None = None
    processor_subfolder: str | None = None


def build_boogu_t2i_messages(prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_T2I}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def build_boogu_edit_messages(prompt: str, image) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_DROP}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
    ]


def build_boogu_drop_messages(prompt: str = "") -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_DROP}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def build_boogu_qwen3vl_8b_config():
    from transformers import Qwen3VLConfig

    config_dict = copy.deepcopy(QWEN3_VL_4B_INSTRUCT_CONFIG)
    config_dict["text_config"].update(
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        num_hidden_layers=36,
        num_key_value_heads=8,
    )
    config_dict["vision_config"].update(
        depth=27,
        hidden_size=1152,
        intermediate_size=4304,
        out_hidden_size=4096,
    )
    return Qwen3VLConfig.from_dict(config_dict)


def _comfy_qwen3vl_key_to_transformers(key: str) -> str:
    if key.startswith("model.language_model.") or key.startswith("model.visual."):
        return key
    if key.startswith("visual."):
        return "model.visual." + key[len("visual.") :]
    if key.startswith("language_model."):
        return "model." + key
    if key.startswith("model."):
        return "model.language_model." + key[len("model.") :]
    return key


def convert_comfy_qwen3vl_8b_state_dict(state_dict: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.endswith(".comfy_quant") or key.endswith(".weight_scale"):
            continue

        mapped_key = _comfy_qwen3vl_key_to_transformers(key)
        value = tensor
        if key.endswith(".weight"):
            scale_key = key[: -len(".weight")] + ".weight_scale"
            if scale_key in state_dict:
                converted[mapped_key[: -len(".weight")] + ".scale_weight"] = state_dict[scale_key].to(dtype=dtype)
        if value.is_floating_point() and value.dtype not in {
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        }:
            value = value.to(dtype=dtype)
        converted[mapped_key] = value
    return converted


def resolve_boogu_text_encoder_load_plan(
    path: str,
    subfolder: str | None = "auto",
    processor_path: str | None = None,
) -> BooguTextEncoderLoadPlan:
    if path.endswith(".safetensors"):
        return BooguTextEncoderLoadPlan(
            kind="comfy_qwen3vl_8b",
            model_path=path,
            processor_path=processor_path or QWEN3_VL_8B_INSTRUCT_REPO_ID,
        )

    model_subfolder = None if subfolder in {None, "", "auto"} else subfolder
    processor_subfolder = None
    if subfolder == "auto" and os.path.isdir(os.path.join(path, "mllm")):
        model_subfolder = "mllm"
    if model_subfolder == "mllm":
        processor_subfolder = "processor"

    return BooguTextEncoderLoadPlan(
        kind="diffusers_repo",
        model_path=path,
        processor_path=processor_path or path,
        model_subfolder=model_subfolder,
        processor_subfolder=processor_subfolder,
    )


def encode_and_save_batch(
    processor,
    text_encoder,
    batch: list[ItemInfo],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    max_length: int = 1024,
):
    for i, item in enumerate(batch):
        print(f"Item {i}: {item.item_key}, prompt: {item.caption}")
        messages = build_boogu_t2i_messages(item.caption)
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
        with torch.no_grad():
            output = text_encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        valid = inputs["attention_mask"][0].to(dtype=torch.bool)
        embed = output.last_hidden_state[0][valid].to(dtype=dtype).cpu()
        save_text_encoder_output_cache_boogu_image(item, embed)


def boogu_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3-VL text encoder path")
    parser.add_argument(
        "--processor",
        type=str,
        default=None,
        help="Optional Qwen3-VL processor/tokenizer path. Required for single-file ComfyUI text encoder weights.",
    )
    parser.add_argument(
        "--text_encoder_subfolder",
        type=str,
        default="auto",
        help="Qwen3-VL model subfolder. Use 'auto' for Boogu HF repos or single-file ComfyUI text encoders.",
    )
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="data type for the text encoder, default is bfloat16")
    parser.add_argument("--max_text_length", type=int, default=1024, help="Maximum Qwen3-VL token length before truncation")
    return parser


def _load_comfy_qwen3vl_8b_text_encoder(path: str, dtype: torch.dtype, device: torch.device):
    from transformers import Qwen3VLForConditionalGeneration
    from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch

    if dtype in {
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    }:
        logger.warning("ComfyUI Qwen3-VL-8B weights are already fp8-scaled; using bfloat16 for scales and non-fp8 tensors.")
        dtype = torch.bfloat16

    config = build_boogu_qwen3vl_8b_config()
    with init_empty_weights():
        text_encoder = Qwen3VLForConditionalGeneration._from_config(config)

    logger.info(f"Loading ComfyUI Qwen3-VL-8B fp8-scaled instruction encoder from {path}")
    state_dict = load_split_weights(path, device="cpu", disable_mmap=True)
    state_dict = convert_comfy_qwen3vl_8b_state_dict(state_dict, dtype=dtype)
    apply_fp8_monkey_patch(text_encoder, state_dict, use_scaled_mm=False)
    info = text_encoder.load_state_dict(state_dict, strict=True, assign=True)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(
            "ComfyUI Qwen3-VL-8B text encoder checkpoint did not match the model: "
            f"missing={info.missing_keys[:10]}, unexpected={info.unexpected_keys[:10]}"
        )
    text_encoder.to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return text_encoder.model


def load_boogu_text_encoder(
    path: str,
    dtype: torch.dtype,
    device: torch.device,
    subfolder: str | None = "auto",
    processor_path: str | None = None,
):
    from transformers import AutoModel, AutoProcessor

    plan = resolve_boogu_text_encoder_load_plan(path, subfolder=subfolder, processor_path=processor_path)
    processor_kwargs = {}
    if plan.processor_subfolder is not None:
        processor_kwargs["subfolder"] = plan.processor_subfolder
    processor = AutoProcessor.from_pretrained(plan.processor_path, **processor_kwargs)

    if plan.kind == "comfy_qwen3vl_8b":
        text_encoder = _load_comfy_qwen3vl_8b_text_encoder(plan.model_path, dtype, device)
    else:
        model_kwargs = {}
        if plan.model_subfolder is not None:
            model_kwargs["subfolder"] = plan.model_subfolder
        text_encoder = AutoModel.from_pretrained(plan.model_path, torch_dtype=dtype, **model_kwargs)
        text_encoder.to(device)

    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return processor, text_encoder


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = boogu_image_setup_parser(parser)
    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    te_dtype = torch.bfloat16
    if args.text_encoder_dtype is not None:
        from musubi_tuner.utils.model_utils import str_to_dtype

        te_dtype = str_to_dtype(args.text_encoder_dtype)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_BOOGU_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    logger.info(f"Loading Qwen3-VL instruction encoder from {args.text_encoder}")
    processor, text_encoder_model = load_boogu_text_encoder(
        args.text_encoder,
        te_dtype,
        device,
        subfolder=args.text_encoder_subfolder,
        processor_path=args.processor,
    )

    def encode_for_text_encoder(batch: list[ItemInfo], _processor=processor, _text_encoder=text_encoder_model):
        encode_and_save_batch(_processor, _text_encoder, batch, device, dtype=te_dtype, max_length=args.max_text_length)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )
    del text_encoder_model

    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


if __name__ == "__main__":
    main()
