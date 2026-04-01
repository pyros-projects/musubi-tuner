#!/usr/bin/env python3
"""Caption images with Qwen vision-language models from a TOML config.

Supports:
- Qwen/Qwen3.5-2B
- Qwen3-VL local or Hub checkpoints
- Qwen2.5-VL local or Hub checkpoints

Example:
    python tools/qwen_vl_caption.py examples/qwen_vl_caption.toml
    python tools/qwen_vl_caption.py examples/qwen_vl_caption.toml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

DEFAULT_SEARCH_PATHS = [
    Path.home() / "repos/ComfyUI/models/LLM/Qwen-VL",
]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


@dataclass(frozen=True)
class ModelSettings:
    source: str
    device: str
    dtype: str
    quantization: str
    attn_implementation: str
    trust_remote_code: bool
    local_files_only: bool
    min_pixels: int | None
    max_pixels: int | None


@dataclass(frozen=True)
class InputSettings:
    path: Path
    recursive: bool
    extensions: tuple[str, ...]
    max_images: int | None
    relative_root: Path


@dataclass(frozen=True)
class PromptSettings:
    instruction: str
    system_prompt: str | None
    strip_think: bool
    strip_code_fences: bool
    collapse_whitespace: bool
    trim_whitespace: bool


@dataclass(frozen=True)
class GenerationSettings:
    batch_size: int
    max_new_tokens: int
    enable_thinking: bool | None
    do_sample: bool
    temperature: float
    top_p: float
    num_beams: int
    repetition_penalty: float
    seed: int


@dataclass(frozen=True)
class OutputSettings:
    format: str
    txt_root: Path | None
    jsonl_path: Path | None
    overwrite: bool
    include_metadata: bool
    prepend: str | None
    append: str | None


@dataclass(frozen=True)
class RunSettings:
    mode: str


@dataclass(frozen=True)
class ConstantSettings:
    content: str
    collapse_whitespace: bool
    trim_whitespace: bool


@dataclass(frozen=True)
class ReplacementRule:
    find: str
    replace: str


@dataclass(frozen=True)
class RegexReplacementRule:
    pattern: str
    replace: str
    ignore_case: bool


@dataclass(frozen=True)
class PostprocessTargetSettings:
    path: Path
    kind: str
    recursive: bool
    extensions: tuple[str, ...]
    max_files: int | None
    relative_root: Path
    jsonl_field: str


@dataclass(frozen=True)
class PostprocessSettings:
    strip_think: bool
    strip_code_fences: bool
    collapse_whitespace: bool
    trim_whitespace: bool
    lowercase: bool
    remove_substrings: tuple[str, ...]
    prepend: str | None
    append: str | None
    ensure_prefix: str | None
    ensure_suffix: str | None
    replacements: tuple[ReplacementRule, ...]
    regex_replacements: tuple[RegexReplacementRule, ...]
    tag_separator: str | None
    dedupe_tags: bool
    sort_tags: bool
    add_tags: tuple[str, ...]
    remove_tags: tuple[str, ...]


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(json.dumps(toml.load(handle)))


def _parse_run_settings(cfg: dict[str, Any]) -> RunSettings:
    run_cfg = cfg.get("run") or {}
    mode = (run_cfg.get("mode", "caption") or "caption").strip().lower()
    if mode not in {"caption", "postprocess", "constant"}:
        raise ValueError("run.mode must be one of: caption, postprocess, constant")
    return RunSettings(mode=mode)


def _read_optional_text(value: str | None, file_value: str | None, label: str) -> str | None:
    if value and file_value:
        raise ValueError(f"Use either `{label}` or `{label}_file`, not both")
    if file_value:
        return Path(file_value).read_text(encoding="utf-8").strip()
    if value is not None:
        return value.strip()
    return None


def _resolve_device(device: str) -> str:
    normalized = (device or "auto").strip()
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return normalized


def _resolve_dtype(dtype_name: str) -> torch.dtype | None:
    normalized = (dtype_name or "auto").strip().lower()
    mapping = {
        "auto": None,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    return mapping[normalized]


def _resolve_attn_implementation(mode: str, device: str, force_sdpa: bool) -> str:
    normalized = (mode or "auto").strip()
    if force_sdpa:
        return "sdpa"
    if normalized != "auto":
        return normalized
    if device.startswith("cuda"):
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except Exception:
            return "sdpa"
    return "sdpa"


def _build_quantization_config(mode: str) -> BitsAndBytesConfig | None:
    normalized = (mode or "none").strip().lower()
    if normalized == "none":
        return None
    if normalized == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    if normalized == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization {mode!r}")


def _detect_prequantized_fp8(source: str) -> bool:
    source_path = Path(source)
    if not source_path.exists():
        return "-fp8" in source.lower() or "_fp8" in source.lower()

    config_path = source_path / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    quant_cfg = config.get("quantization_config") or {}
    return isinstance(quant_cfg, dict) and quant_cfg.get("fmt") == "e4m3"


def _resolve_model_source(model_cfg: dict[str, Any]) -> str:
    path = model_cfg.get("path")
    repo_id = model_cfg.get("repo_id")
    source = model_cfg.get("source")
    name = model_cfg.get("name")

    specified = [item for item in (path, repo_id, source, name) if item]
    if len(specified) != 1:
        raise ValueError("Specify exactly one of model.path, model.repo_id, model.source, or model.name")

    if path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Model path does not exist: {resolved}")
        return str(resolved)

    if repo_id:
        return str(repo_id)

    if source:
        source_path = Path(source).expanduser()
        if source_path.exists():
            return str(source_path.resolve())
        if "/" in source:
            return source
        name = source

    assert name is not None
    search_paths = [Path(p).expanduser() for p in model_cfg.get("search_paths", [])]
    search_paths.extend(path for path in DEFAULT_SEARCH_PATHS if path.exists())
    for root in search_paths:
        candidate = root / name
        if candidate.exists():
            return str(candidate.resolve())
    if "/" not in name:
        return f"Qwen/{name}"
    return name


def _parse_model_settings(cfg: dict[str, Any]) -> ModelSettings:
    model_cfg = cfg.get("model") or {}
    source = _resolve_model_source(model_cfg)
    device = _resolve_device(model_cfg.get("device", "auto"))
    quantization = (model_cfg.get("quantization", "none") or "none").strip().lower()
    min_pixels = model_cfg.get("min_pixels")
    max_pixels = model_cfg.get("max_pixels")
    return ModelSettings(
        source=source,
        device=device,
        dtype=model_cfg.get("dtype", "auto"),
        quantization=quantization,
        attn_implementation=model_cfg.get("attn_implementation", "auto"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def _parse_input_settings(cfg: dict[str, Any]) -> InputSettings:
    input_cfg = cfg.get("input") or {}
    path_value = input_cfg.get("path")
    if not path_value:
        raise ValueError("Missing required input.path")

    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    extensions = tuple(
        ext if ext.startswith(".") else f".{ext}" for ext in (input_cfg.get("extensions") or sorted(IMAGE_EXTENSIONS))
    )
    relative_root = Path(input_cfg.get("relative_root", str(path if path.is_dir() else path.parent)))
    relative_root = relative_root.expanduser().resolve()
    return InputSettings(
        path=path,
        recursive=bool(input_cfg.get("recursive", True)),
        extensions=tuple(ext.lower() for ext in extensions),
        max_images=input_cfg.get("max_images"),
        relative_root=relative_root,
    )


def _parse_prompt_settings(cfg: dict[str, Any]) -> PromptSettings:
    prompt_cfg = cfg.get("prompt") or {}
    instruction = _read_optional_text(
        prompt_cfg.get("instruction"),
        prompt_cfg.get("instruction_file"),
        "prompt.instruction",
    )
    if not instruction:
        raise ValueError("Missing required prompt.instruction (or prompt.instruction_file)")
    system_prompt = _read_optional_text(
        prompt_cfg.get("system_prompt"),
        prompt_cfg.get("system_prompt_file"),
        "prompt.system_prompt",
    )
    return PromptSettings(
        instruction=instruction,
        system_prompt=system_prompt,
        strip_think=bool(prompt_cfg.get("strip_think", True)),
        strip_code_fences=bool(prompt_cfg.get("strip_code_fences", True)),
        collapse_whitespace=bool(prompt_cfg.get("collapse_whitespace", True)),
        trim_whitespace=bool(prompt_cfg.get("trim_whitespace", True)),
    )


def _parse_generation_settings(cfg: dict[str, Any]) -> GenerationSettings:
    generation_cfg = cfg.get("generation") or {}
    enable_thinking = generation_cfg.get("enable_thinking")
    if enable_thinking is not None and not isinstance(enable_thinking, bool):
        raise ValueError("generation.enable_thinking must be a boolean when set")
    return GenerationSettings(
        batch_size=int(generation_cfg.get("batch_size", 1)),
        max_new_tokens=int(generation_cfg.get("max_new_tokens", 256)),
        enable_thinking=enable_thinking,
        do_sample=bool(generation_cfg.get("do_sample", False)),
        temperature=float(generation_cfg.get("temperature", 0.2)),
        top_p=float(generation_cfg.get("top_p", 0.9)),
        num_beams=int(generation_cfg.get("num_beams", 1)),
        repetition_penalty=float(generation_cfg.get("repetition_penalty", 1.05)),
        seed=int(generation_cfg.get("seed", 1)),
    )


def _parse_output_settings(cfg: dict[str, Any], config_path: Path) -> OutputSettings:
    output_cfg = cfg.get("output") or {}
    output_format = (output_cfg.get("format", "txt") or "txt").strip().lower()
    if output_format not in {"txt", "jsonl", "both"}:
        raise ValueError("output.format must be one of: txt, jsonl, both")

    txt_root = output_cfg.get("txt_root")
    jsonl_path = output_cfg.get("jsonl_path")

    resolved_txt_root = Path(txt_root).expanduser().resolve() if txt_root else None
    if output_format in {"jsonl", "both"}:
        if jsonl_path:
            resolved_jsonl_path = Path(jsonl_path).expanduser().resolve()
        else:
            resolved_jsonl_path = config_path.with_suffix(".captions.jsonl")
    else:
        resolved_jsonl_path = None

    return OutputSettings(
        format=output_format,
        txt_root=resolved_txt_root,
        jsonl_path=resolved_jsonl_path,
        overwrite=bool(output_cfg.get("overwrite", False)),
        include_metadata=bool(output_cfg.get("include_metadata", True)),
        prepend=_read_optional_text(
            output_cfg.get("prepend"),
            output_cfg.get("prepend_file"),
            "output.prepend",
        ),
        append=_read_optional_text(
            output_cfg.get("append"),
            output_cfg.get("append_file"),
            "output.append",
        ),
    )


def _parse_postprocess_target_settings(cfg: dict[str, Any]) -> PostprocessTargetSettings:
    input_cfg = cfg.get("input") or {}
    path_value = input_cfg.get("path")
    if not path_value:
        raise ValueError("Missing required input.path")

    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file() and path.suffix.lower() == ".jsonl":
        return PostprocessTargetSettings(
            path=path,
            kind="jsonl",
            recursive=False,
            extensions=(".jsonl",),
            max_files=1,
            relative_root=path.parent,
            jsonl_field=str(input_cfg.get("jsonl_field", "caption")),
        )

    extensions = tuple(ext if ext.startswith(".") else f".{ext}" for ext in (input_cfg.get("extensions") or [".txt"]))
    relative_root = Path(input_cfg.get("relative_root", str(path if path.is_dir() else path.parent)))
    relative_root = relative_root.expanduser().resolve()
    return PostprocessTargetSettings(
        path=path,
        kind="txt",
        recursive=bool(input_cfg.get("recursive", True)),
        extensions=tuple(ext.lower() for ext in extensions),
        max_files=input_cfg.get("max_files", input_cfg.get("max_images")),
        relative_root=relative_root,
        jsonl_field=str(input_cfg.get("jsonl_field", "caption")),
    )


def _parse_postprocess_settings(cfg: dict[str, Any]) -> PostprocessSettings:
    pp_cfg = cfg.get("postprocess") or {}

    replacements = []
    for item in pp_cfg.get("replace", []):
        find = item.get("find")
        if not find:
            raise ValueError("Each [[postprocess.replace]] entry needs `find`")
        replacements.append(
            ReplacementRule(
                find=str(find),
                replace=str(item.get("replace", "")),
            )
        )

    regex_replacements = []
    for item in pp_cfg.get("regex_replace", []):
        pattern = item.get("pattern")
        if not pattern:
            raise ValueError("Each [[postprocess.regex_replace]] entry needs `pattern`")
        regex_replacements.append(
            RegexReplacementRule(
                pattern=str(pattern),
                replace=str(item.get("replace", "")),
                ignore_case=bool(item.get("ignore_case", False)),
            )
        )

    tag_cfg = pp_cfg.get("tags") or {}
    tag_separator = tag_cfg.get("separator")
    if tag_separator is not None:
        tag_separator = str(tag_separator)

    return PostprocessSettings(
        strip_think=bool(pp_cfg.get("strip_think", False)),
        strip_code_fences=bool(pp_cfg.get("strip_code_fences", False)),
        collapse_whitespace=bool(pp_cfg.get("collapse_whitespace", True)),
        trim_whitespace=bool(pp_cfg.get("trim_whitespace", True)),
        lowercase=bool(pp_cfg.get("lowercase", False)),
        remove_substrings=tuple(str(item) for item in pp_cfg.get("remove_substrings", [])),
        prepend=_read_optional_text(
            pp_cfg.get("prepend"),
            pp_cfg.get("prepend_file"),
            "postprocess.prepend",
        ),
        append=_read_optional_text(
            pp_cfg.get("append"),
            pp_cfg.get("append_file"),
            "postprocess.append",
        ),
        ensure_prefix=_read_optional_text(
            pp_cfg.get("ensure_prefix"),
            pp_cfg.get("ensure_prefix_file"),
            "postprocess.ensure_prefix",
        ),
        ensure_suffix=_read_optional_text(
            pp_cfg.get("ensure_suffix"),
            pp_cfg.get("ensure_suffix_file"),
            "postprocess.ensure_suffix",
        ),
        replacements=tuple(replacements),
        regex_replacements=tuple(regex_replacements),
        tag_separator=tag_separator,
        dedupe_tags=bool(tag_cfg.get("dedupe", False)),
        sort_tags=bool(tag_cfg.get("sort", False)),
        add_tags=tuple(str(item) for item in tag_cfg.get("add", [])),
        remove_tags=tuple(str(item) for item in tag_cfg.get("remove", [])),
    )


def _parse_constant_settings(cfg: dict[str, Any]) -> ConstantSettings:
    constant_cfg = cfg.get("constant") or {}
    content = _read_optional_text(
        constant_cfg.get("content"),
        constant_cfg.get("content_file"),
        "constant.content",
    )
    if content is None:
        raise ValueError("Missing required constant.content (or constant.content_file)")

    if bool(constant_cfg.get("collapse_whitespace", False)):
        content = " ".join(content.split())
    if bool(constant_cfg.get("trim_whitespace", True)):
        content = content.strip()
    if not content:
        raise ValueError("constant.content must not be empty after cleanup")

    return ConstantSettings(
        content=content,
        collapse_whitespace=bool(constant_cfg.get("collapse_whitespace", False)),
        trim_whitespace=bool(constant_cfg.get("trim_whitespace", True)),
    )


def _iter_images(settings: InputSettings) -> list[Path]:
    if settings.path.is_file():
        candidates = [settings.path]
    else:
        glob_pattern = "**/*" if settings.recursive else "*"
        candidates = sorted(path for path in settings.path.glob(glob_pattern) if path.is_file())

    images = [path for path in candidates if path.suffix.lower() in settings.extensions]
    if settings.max_images is not None:
        images = images[: settings.max_images]
    if not images:
        raise ValueError(f"No images found under {settings.path}")
    return images


def _strip_code_fences(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            continue
        if stripped == "```":
            continue
        lines.append(line)
    return "\n".join(lines)


def _strip_think_blocks(text: str) -> str:
    while True:
        start = text.lower().find("<think")
        if start == -1:
            break
        end = text.lower().find("</think>", start)
        if end == -1:
            text = text[:start]
            break
        text = text[:start] + text[end + len("</think>") :]
    return text


def _cleanup_text(
    text: str,
    *,
    strip_think: bool,
    strip_code_fences: bool,
    collapse_whitespace: bool,
    trim_whitespace: bool,
) -> str:
    cleaned = text or ""
    if strip_think:
        cleaned = _strip_think_blocks(cleaned)
    if strip_code_fences:
        cleaned = _strip_code_fences(cleaned)
    if collapse_whitespace:
        cleaned = " ".join(cleaned.split())
    if trim_whitespace:
        cleaned = cleaned.strip()
    return cleaned


def _clean_caption(text: str, prompt_settings: PromptSettings) -> str:
    return _cleanup_text(
        text,
        strip_think=prompt_settings.strip_think,
        strip_code_fences=prompt_settings.strip_code_fences,
        collapse_whitespace=prompt_settings.collapse_whitespace,
        trim_whitespace=prompt_settings.trim_whitespace,
    )


def _txt_output_path(image_path: Path, image_settings: InputSettings, output_settings: OutputSettings) -> Path:
    if output_settings.txt_root is None:
        return image_path.with_suffix(".txt")
    relative = image_path.relative_to(image_settings.relative_root)
    return (output_settings.txt_root / relative).with_suffix(".txt")


def _load_existing_jsonl_records(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            rel_path = payload.get("relative_path")
            if isinstance(rel_path, str):
                seen.add(rel_path)
    return seen


class CaptionRunner:
    def __init__(
        self,
        model_settings: ModelSettings,
        prompt_settings: PromptSettings,
        generation_settings: GenerationSettings,
    ) -> None:
        self.model_settings = model_settings
        self.prompt_settings = prompt_settings
        self.generation_settings = generation_settings
        self.model = None
        self.processor = None
        self.model_device = None

    def _build_conversation_prompt(self, image: Image.Image) -> str:
        assert self.processor is not None

        conversation: list[dict[str, Any]] = []
        if self.prompt_settings.system_prompt:
            conversation.append({"role": "system", "content": self.prompt_settings.system_prompt})
        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt_settings.instruction},
                ],
            }
        )
        chat_template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.generation_settings.enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = self.generation_settings.enable_thinking
        try:
            return self.processor.apply_chat_template(conversation, **chat_template_kwargs)
        except TypeError as exc:
            if "enable_thinking" not in chat_template_kwargs:
                raise
            raise RuntimeError(
                "generation.enable_thinking is set, but this processor/chat template "
                "does not accept enable_thinking. Update transformers/model files or "
                "remove generation.enable_thinking for this model."
            ) from exc

    def load(self) -> None:
        source = self.model_settings.source
        requested_device = self.model_settings.device
        dtype = _resolve_dtype(self.model_settings.dtype)
        prequantized_fp8 = _detect_prequantized_fp8(source)
        quantization_config = _build_quantization_config(self.model_settings.quantization)
        attn_implementation = _resolve_attn_implementation(
            self.model_settings.attn_implementation,
            requested_device,
            force_sdpa=prequantized_fp8 or quantization_config is not None,
        )
        resolved_device = _resolve_device(requested_device)

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.model_settings.trust_remote_code,
            "local_files_only": self.model_settings.local_files_only,
            "attn_implementation": attn_implementation,
        }

        if prequantized_fp8:
            if dtype is not None:
                load_kwargs["dtype"] = dtype
            self.model = AutoModelForImageTextToText.from_pretrained(source, **load_kwargs).eval()
            self.model = self.model.to(resolved_device)
        else:
            load_kwargs["device_map"] = "auto" if requested_device == "auto" else resolved_device
            load_kwargs["low_cpu_mem_usage"] = True
            if dtype is not None:
                load_kwargs["dtype"] = dtype
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
            if load_kwargs["device_map"] is None:
                load_kwargs.pop("device_map")
            self.model = AutoModelForImageTextToText.from_pretrained(source, **load_kwargs).eval()
            if resolved_device == "cpu":
                self.model = self.model.to("cpu")

        self.processor = AutoProcessor.from_pretrained(
            source,
            trust_remote_code=self.model_settings.trust_remote_code,
            local_files_only=self.model_settings.local_files_only,
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        if self.model_settings.min_pixels is not None:
            setattr(self.processor.image_processor, "min_pixels", self.model_settings.min_pixels)
        if self.model_settings.max_pixels is not None:
            setattr(self.processor.image_processor, "max_pixels", self.model_settings.max_pixels)

        self.model_device = getattr(self.model, "device", None)
        if self.model_device is None:
            self.model_device = next(self.model.parameters()).device

    def caption_image(self, image_path: Path) -> str:
        return self.caption_images([image_path])[0]

    def caption_images(self, image_paths: list[Path]) -> list[str]:
        assert self.model is not None
        assert self.processor is not None

        prompts: list[str] = []
        images: list[Image.Image] = []
        handles: list[Image.Image] = []
        try:
            for image_path in image_paths:
                handle = Image.open(image_path)
                handles.append(handle)
                pil_image = handle.convert("RGB")
                prompts.append(self._build_conversation_prompt(pil_image))
                images.append(pil_image)

            processed = self.processor(
                text=prompts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
        finally:
            for handle in handles:
                handle.close()

        inputs = {key: value.to(self.model_device) if torch.is_tensor(value) else value for key, value in processed.items()}

        torch.manual_seed(self.generation_settings.seed)
        generate_kwargs = {
            "max_new_tokens": self.generation_settings.max_new_tokens,
            "num_beams": self.generation_settings.num_beams,
            "repetition_penalty": self.generation_settings.repetition_penalty,
        }
        if self.generation_settings.num_beams > 1:
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = self.generation_settings.do_sample
            if self.generation_settings.do_sample:
                generate_kwargs["temperature"] = self.generation_settings.temperature
                generate_kwargs["top_p"] = self.generation_settings.top_p

        output_ids = self.model.generate(**inputs, **generate_kwargs)
        prompt_ids = inputs["input_ids"]
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(prompt_ids, output_ids, strict=False)]
        captions = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [_clean_caption(caption, self.prompt_settings) for caption in captions]


def _build_json_record(
    image_path: Path,
    image_settings: InputSettings,
    caption: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "image_path": str(image_path),
        "relative_path": str(image_path.relative_to(image_settings.relative_root)),
        "caption": caption,
    }
    with Image.open(image_path) as image:
        record["width"] = image.width
        record["height"] = image.height
    if extra_metadata:
        record.update(extra_metadata)
    return record


def _write_txt(path: Path, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def _apply_output_affixes(caption: str, output_settings: OutputSettings) -> str:
    text = caption
    if output_settings.prepend:
        separator = "" if output_settings.prepend[-1].isspace() else " "
        text = f"{output_settings.prepend}{separator}{text}"
    if output_settings.append:
        separator = ""
        if not output_settings.append[0].isspace() and output_settings.append[0] not in ",.;:!?)]}":
            separator = " "
        text = f"{text}{separator}{output_settings.append}"
    return " ".join(text.split())


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prepare_caption_work_items(
    images: list[Path],
    input_settings: InputSettings,
    output_settings: OutputSettings,
    existing_jsonl: set[str],
) -> tuple[list[dict[str, Any]], int]:
    skipped_count = 0
    work_items: list[dict[str, Any]] = []

    for image_path in images:
        relative_path = str(image_path.relative_to(input_settings.relative_root))

        txt_path = None
        should_skip_txt = False
        if output_settings.format in {"txt", "both"}:
            txt_path = _txt_output_path(image_path, input_settings, output_settings)
            should_skip_txt = txt_path.exists() and not output_settings.overwrite

        should_skip_jsonl = (
            output_settings.format in {"jsonl", "both"} and not output_settings.overwrite and relative_path in existing_jsonl
        )

        required_outputs_present = []
        if output_settings.format in {"txt", "both"}:
            required_outputs_present.append(should_skip_txt)
        if output_settings.format in {"jsonl", "both"}:
            required_outputs_present.append(should_skip_jsonl)

        if required_outputs_present and all(required_outputs_present):
            skipped_count += 1
            continue

        work_items.append(
            {
                "image_path": image_path,
                "relative_path": relative_path,
                "txt_path": txt_path,
                "should_skip_txt": should_skip_txt,
                "should_skip_jsonl": should_skip_jsonl,
            }
        )

    return work_items, skipped_count


def _write_caption_outputs(
    item: dict[str, Any],
    caption: str,
    input_settings: InputSettings,
    output_settings: OutputSettings,
    existing_jsonl: set[str],
    json_record_extra_metadata: dict[str, Any] | None = None,
) -> None:
    caption = _apply_output_affixes(caption, output_settings)

    if item["txt_path"] is not None and not item["should_skip_txt"]:
        _write_txt(item["txt_path"], caption)

    if output_settings.format in {"jsonl", "both"} and not item["should_skip_jsonl"]:
        record = _build_json_record(
            item["image_path"],
            input_settings,
            caption,
            extra_metadata=json_record_extra_metadata,
        )
        if not output_settings.include_metadata:
            record = {
                "image_path": record["image_path"],
                "relative_path": record["relative_path"],
                "caption": record["caption"],
            }
        _append_jsonl(output_settings.jsonl_path, record)
        existing_jsonl.add(item["relative_path"])


def inspect_caption_run(config_path: Path) -> dict[str, Any]:
    cfg = _load_toml(config_path)
    run_settings = _parse_run_settings(cfg)
    if run_settings.mode != "caption":
        raise ValueError(f"inspect_caption_run only supports caption mode, got {run_settings.mode!r}")

    input_settings = _parse_input_settings(cfg)
    output_settings = _parse_output_settings(cfg, config_path)
    images = _iter_images(input_settings)

    existing_jsonl = (
        _load_existing_jsonl_records(output_settings.jsonl_path)
        if output_settings.jsonl_path and not output_settings.overwrite
        else set()
    )
    work_items, skipped_count = _prepare_caption_work_items(
        images,
        input_settings,
        output_settings,
        existing_jsonl,
    )
    return {
        "total_images": len(images),
        "to_process": len(work_items),
        "to_skip": skipped_count,
    }


def _iter_postprocess_files(settings: PostprocessTargetSettings) -> list[Path]:
    if settings.kind == "jsonl":
        return [settings.path]

    if settings.path.is_file():
        candidates = [settings.path]
    else:
        glob_pattern = "**/*" if settings.recursive else "*"
        candidates = sorted(path for path in settings.path.glob(glob_pattern) if path.is_file())

    files = [path for path in candidates if path.suffix.lower() in settings.extensions]
    if settings.max_files is not None:
        files = files[: settings.max_files]
    if not files:
        raise ValueError(f"No matching caption files found under {settings.path}")
    return files


def _apply_tag_operations(text: str, settings: PostprocessSettings) -> str:
    has_tag_ops = any(
        (
            settings.tag_separator,
            settings.dedupe_tags,
            settings.sort_tags,
            settings.add_tags,
            settings.remove_tags,
        )
    )
    if not has_tag_ops:
        return text

    separator = settings.tag_separator or ","
    raw_tags = [item.strip() for item in text.split(separator)]
    tags = [item for item in raw_tags if item]
    remove_set = {item.strip() for item in settings.remove_tags if item.strip()}

    filtered = []
    for tag in tags:
        if tag in remove_set:
            continue
        filtered.append(tag)

    for tag in settings.add_tags:
        normalized = tag.strip()
        if normalized:
            filtered.append(normalized)

    if settings.dedupe_tags:
        deduped = []
        seen = set()
        for tag in filtered:
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(tag)
        filtered = deduped

    if settings.sort_tags:
        filtered = sorted(filtered, key=str.casefold)

    return f"{separator} ".join(filtered)


def _postprocess_text(text: str, settings: PostprocessSettings) -> str:
    updated = text

    if settings.strip_think or settings.strip_code_fences:
        updated = _cleanup_text(
            updated,
            strip_think=settings.strip_think,
            strip_code_fences=settings.strip_code_fences,
            collapse_whitespace=False,
            trim_whitespace=False,
        )

    for snippet in settings.remove_substrings:
        updated = updated.replace(snippet, "")

    for rule in settings.replacements:
        updated = updated.replace(rule.find, rule.replace)

    for rule in settings.regex_replacements:
        flags = re.IGNORECASE if rule.ignore_case else 0
        updated = re.sub(rule.pattern, rule.replace, updated, flags=flags)

    if settings.ensure_prefix and not updated.startswith(settings.ensure_prefix):
        updated = settings.ensure_prefix + updated
    if settings.ensure_suffix and not updated.endswith(settings.ensure_suffix):
        updated = updated + settings.ensure_suffix

    if settings.prepend:
        updated = settings.prepend + updated
    if settings.append:
        updated = updated + settings.append

    if settings.lowercase:
        updated = updated.lower()

    updated = _apply_tag_operations(updated, settings)

    updated = _cleanup_text(
        updated,
        strip_think=False,
        strip_code_fences=False,
        collapse_whitespace=settings.collapse_whitespace,
        trim_whitespace=settings.trim_whitespace,
    )
    return updated


def _run_postprocess(
    config_path: Path,
    target_settings: PostprocessTargetSettings,
    postprocess_settings: PostprocessSettings,
    dry_run: bool,
) -> None:
    targets = _iter_postprocess_files(target_settings)

    if dry_run:
        print(f"config: {config_path}")
        print("mode: postprocess")
        print(f"kind: {target_settings.kind}")
        print(f"targets: {len(targets)}")
        print(f"path: {target_settings.path}")
        if target_settings.kind == "jsonl":
            print(f"jsonl_field: {target_settings.jsonl_field}")
        return

    changed_count = 0
    unchanged_count = 0

    if target_settings.kind == "jsonl":
        path = targets[0]
        rewritten_lines = []
        total_records = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                total_records += 1
                stripped = line.rstrip("\n")
                if not stripped:
                    rewritten_lines.append(line)
                    unchanged_count += 1
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    rewritten_lines.append(line)
                    unchanged_count += 1
                    continue
                original = payload.get(target_settings.jsonl_field)
                if not isinstance(original, str):
                    rewritten_lines.append(json.dumps(payload, ensure_ascii=False) + "\n")
                    unchanged_count += 1
                    continue
                updated = _postprocess_text(original, postprocess_settings)
                if updated != original:
                    payload[target_settings.jsonl_field] = updated
                    changed_count += 1
                else:
                    unchanged_count += 1
                rewritten_lines.append(json.dumps(payload, ensure_ascii=False) + "\n")
        path.write_text("".join(rewritten_lines), encoding="utf-8")
        print(f"Postprocess complete: changed={changed_count} unchanged={unchanged_count} total={total_records}")
        return

    for path in tqdm(targets, desc="Postprocessing", unit="file"):
        original = path.read_text(encoding="utf-8")
        updated = _postprocess_text(original, postprocess_settings)
        if updated == original:
            unchanged_count += 1
            continue
        path.write_text(updated + ("\n" if original.endswith("\n") else ""), encoding="utf-8")
        changed_count += 1

    print(f"Postprocess complete: changed={changed_count} unchanged={unchanged_count} total={len(targets)}")


def run(config_path: Path, dry_run: bool) -> None:
    cfg = _load_toml(config_path)
    run_settings = _parse_run_settings(cfg)

    if run_settings.mode == "postprocess":
        target_settings = _parse_postprocess_target_settings(cfg)
        postprocess_settings = _parse_postprocess_settings(cfg)
        _run_postprocess(config_path, target_settings, postprocess_settings, dry_run)
        return

    if run_settings.mode == "constant":
        constant_settings = _parse_constant_settings(cfg)
        input_settings = _parse_input_settings(cfg)
        output_settings = _parse_output_settings(cfg, config_path)
        images = _iter_images(input_settings)

        existing_jsonl = (
            _load_existing_jsonl_records(output_settings.jsonl_path)
            if output_settings.jsonl_path and not output_settings.overwrite
            else set()
        )
        work_items, skipped_count = _prepare_caption_work_items(
            images,
            input_settings,
            output_settings,
            existing_jsonl,
        )

        if dry_run:
            print(f"config: {config_path}")
            print("mode: constant")
            print(f"images: {len(images)}")
            print(f"constant_content: {constant_settings.content}")
            print(f"output_format: {output_settings.format}")
            print(f"jsonl_path: {output_settings.jsonl_path}")
            print(f"txt_root: {output_settings.txt_root or '(sidecar)'}")
            print(f"output_prepend: {output_settings.prepend!r}")
            print(f"output_append: {output_settings.append!r}")
            print(f"would_process: {len(work_items)}")
            print(f"would_skip: {skipped_count}")
            return

        if output_settings.overwrite and output_settings.jsonl_path and output_settings.jsonl_path.exists():
            output_settings.jsonl_path.unlink()

        processed_count = 0
        json_record_extra_metadata = {
            "content_source": "constant",
            "constant_content": constant_settings.content,
            "output_prepend": output_settings.prepend,
            "output_append": output_settings.append,
        }
        for item in tqdm(work_items, desc="Writing", unit="image"):
            _write_caption_outputs(
                item,
                constant_settings.content,
                input_settings,
                output_settings,
                existing_jsonl,
                json_record_extra_metadata=json_record_extra_metadata,
            )
            processed_count += 1

        print(f"Constant caption write complete: processed={processed_count} skipped={skipped_count} total={len(images)}")
        return

    model_settings = _parse_model_settings(cfg)
    input_settings = _parse_input_settings(cfg)
    prompt_settings = _parse_prompt_settings(cfg)
    generation_settings = _parse_generation_settings(cfg)
    output_settings = _parse_output_settings(cfg, config_path)
    images = _iter_images(input_settings)

    existing_jsonl = (
        _load_existing_jsonl_records(output_settings.jsonl_path)
        if output_settings.jsonl_path and not output_settings.overwrite
        else set()
    )

    work_items, skipped_count = _prepare_caption_work_items(
        images,
        input_settings,
        output_settings,
        existing_jsonl,
    )

    if dry_run:
        print(f"config: {config_path}")
        print("mode: caption")
        print(f"model: {model_settings.source}")
        print(f"device: {model_settings.device}")
        print(f"images: {len(images)}")
        print(f"batch_size: {generation_settings.batch_size}")
        print(f"enable_thinking: {generation_settings.enable_thinking!r}")
        print(f"output_format: {output_settings.format}")
        print(f"jsonl_path: {output_settings.jsonl_path}")
        print(f"txt_root: {output_settings.txt_root or '(sidecar)'}")
        print(f"output_prepend: {output_settings.prepend!r}")
        print(f"output_append: {output_settings.append!r}")
        print(f"would_process: {len(work_items)}")
        print(f"would_skip: {skipped_count}")
        return

    if output_settings.overwrite and output_settings.jsonl_path and output_settings.jsonl_path.exists():
        output_settings.jsonl_path.unlink()

    if not work_items:
        print("Nothing to do: all caption outputs already exist. Set output.overwrite = true to regenerate them.")
        return

    runner = CaptionRunner(model_settings, prompt_settings, generation_settings)
    runner.load()

    batch_size = max(1, generation_settings.batch_size)
    processed_count = 0
    json_record_extra_metadata = {
        "instruction": prompt_settings.instruction,
        "model_source": model_settings.source,
        "generation_enable_thinking": generation_settings.enable_thinking,
        "output_prepend": output_settings.prepend,
        "output_append": output_settings.append,
    }

    for start in tqdm(range(0, len(work_items), batch_size), desc="Generating", unit="batch"):
        batch = work_items[start : start + batch_size]
        captions = runner.caption_images([item["image_path"] for item in batch])
        if len(captions) != len(batch):
            raise RuntimeError("Batch caption count did not match batch size")

        for item, caption in zip(batch, captions, strict=False):
            if not caption:
                raise RuntimeError(f"Model returned an empty caption for {item['image_path']}")

            _write_caption_outputs(
                item,
                caption,
                input_settings,
                output_settings,
                existing_jsonl,
                json_record_extra_metadata=json_record_extra_metadata,
            )
            processed_count += 1

    print(f"Captioning complete: processed={processed_count} skipped={skipped_count} total={len(images)}")
    if skipped_count:
        print("Skipped existing outputs because output.overwrite = false.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to captioning TOML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and list work only")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    run(config_path, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
