from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml


VALID_LORA_MODES = {"extend", "replace"}
VALID_IMAGE_INPUT_ORDERS = {"sorted", "random"}
VALID_IMAGE_ASSIGNMENTS = {"unique"}
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def validate_ltx_prompt_lora_entry(entry: Any) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("Each LoRA entry must be a TOML table/object.")

    path = entry.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Each LoRA entry must include a non-empty string 'path'.")

    normalized: Dict[str, Any] = {"path": path}

    weight = entry.get("weight", 1.0)
    try:
        normalized["weight"] = float(weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid LoRA weight for {path!r}: {weight!r}") from exc

    merge = entry.get("merge", False)
    if not isinstance(merge, bool):
        raise ValueError(f"Invalid LoRA merge flag for {path!r}: {merge!r}")
    normalized["merge"] = merge

    return normalized


def normalize_ltx_prompt_lora_entries(entries: Any) -> List[Dict[str, Any]]:
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError("Prompt-file 'loras' must be a list of TOML tables.")
    return [validate_ltx_prompt_lora_entry(entry) for entry in entries]


def _normalize_optional_path_string(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Prompt-file '{field_name}' must be a non-empty string when provided.")
    return value.strip()


def _normalize_input_images(entries: Any) -> List[str]:
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError("Prompt-file 'input_images' must be a list of image paths.")

    normalized: List[str] = []
    for idx, entry in enumerate(entries):
        path = _normalize_optional_path_string(entry, f"input_images[{idx}]")
        assert path is not None
        normalized.append(path)
    return normalized


def _normalize_image_input_order(value: Any) -> str:
    if value is None:
        return "sorted"
    if not isinstance(value, str) or value not in VALID_IMAGE_INPUT_ORDERS:
        raise ValueError(
            f"Invalid image_input_order {value!r}. Expected one of {sorted(VALID_IMAGE_INPUT_ORDERS)}."
        )
    return value


def _normalize_image_assignment(value: Any) -> str:
    if value is None:
        return "unique"
    if not isinstance(value, str) or value not in VALID_IMAGE_ASSIGNMENTS:
        raise ValueError(
            f"Invalid image_assignment {value!r}. Expected one of {sorted(VALID_IMAGE_ASSIGNMENTS)}."
        )
    return value


def _normalize_use_image_pool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("Prompt-file 'use_image_pool' must be a boolean when provided.")
    return value


def _collect_prompt_pool_images(
    *,
    input_images: List[str],
    input_images_dir: Optional[str],
    image_input_order: str,
) -> List[str]:
    pool_images: List[str] = []

    for raw_path in input_images:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt-file input image not found: {raw_path}")
        if not path.is_file():
            raise ValueError(f"Prompt-file input image is not a file: {raw_path}")
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Prompt-file input image is not a supported image file: {raw_path}")
        pool_images.append(str(path))

    if input_images_dir is not None:
        pool_dir = Path(input_images_dir)
        if not pool_dir.exists():
            raise FileNotFoundError(f"Prompt-file input_images_dir not found: {input_images_dir}")
        if not pool_dir.is_dir():
            raise ValueError(f"Prompt-file input_images_dir is not a directory: {input_images_dir}")

        for candidate in sorted(pool_dir.rglob("*")):
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                pool_images.append(str(candidate))

    if image_input_order == "sorted":
        pool_images.sort()
    else:
        random.shuffle(pool_images)

    return pool_images


def _resolve_subset_image_assignments(
    subsets: List[Dict[str, Any]],
    *,
    pool_images: List[str],
    has_pool_config: bool,
) -> List[Optional[str]]:
    assignments: List[Optional[str]] = [None] * len(subsets)
    eligible_indices: List[int] = []

    for idx, subset in enumerate(subsets):
        explicit_image_path = _normalize_optional_path_string(subset.get("image_path"), "image_path")
        use_image_pool = _normalize_use_image_pool(subset.get("use_image_pool"))

        if explicit_image_path is not None:
            image_path = Path(explicit_image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Prompt-file explicit image_path not found: {explicit_image_path}")
            if not image_path.is_file():
                raise ValueError(f"Prompt-file explicit image_path is not a file: {explicit_image_path}")
            assignments[idx] = str(image_path)
            continue

        if use_image_pool is False:
            continue

        if use_image_pool is True and not has_pool_config:
            raise ValueError("Prompt subset requested use_image_pool = true, but no root image pool is configured.")

        if has_pool_config:
            eligible_indices.append(idx)

    if eligible_indices and not pool_images:
        raise ValueError(
            "Prompt file image pool under [prompt] resolved 0 supported images, "
            "but at least one subset requires pool assignment."
        )

    if len(pool_images) < len(eligible_indices):
        raise ValueError(
            f"Prompt file image pool under [prompt] resolved {len(pool_images)} images, "
            f"but {len(eligible_indices)} subsets require pool assignment. "
            "Add more images, mark some subsets with use_image_pool = false, or provide explicit image_path values."
        )

    for pool_idx, subset_idx in enumerate(eligible_indices):
        assignments[subset_idx] = pool_images[pool_idx]

    return assignments


def resolve_ltx_prompt_file_data(
    data: Dict[str, Any], *, baseline_loras: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    prompt_root = data.get("prompt")
    if not isinstance(prompt_root, dict):
        raise ValueError("LTX prompt TOML must contain a [prompt] table.")

    subsets = prompt_root.get("subset")
    if not isinstance(subsets, list) or not subsets:
        raise ValueError("LTX prompt TOML must contain at least one [[prompt.subset]] entry.")

    root_prompt = deepcopy(prompt_root)
    root_prompt.pop("subset", None)
    root_baseline_loras = normalize_ltx_prompt_lora_entries(root_prompt.pop("loras", None))
    root_input_images = _normalize_input_images(root_prompt.pop("input_images", None))
    root_input_images_dir = _normalize_optional_path_string(root_prompt.pop("input_images_dir", None), "input_images_dir")
    root_image_input_order = _normalize_image_input_order(root_prompt.pop("image_input_order", None))
    _normalize_image_assignment(root_prompt.pop("image_assignment", None))
    cli_baseline_loras = normalize_ltx_prompt_lora_entries(baseline_loras)
    combined_baseline = [*cli_baseline_loras, *root_baseline_loras]
    has_pool_config = bool(root_input_images or root_input_images_dir)
    pool_images = _collect_prompt_pool_images(
        input_images=root_input_images,
        input_images_dir=root_input_images_dir,
        image_input_order=root_image_input_order,
    )
    image_assignments = _resolve_subset_image_assignments(subsets, pool_images=pool_images, has_pool_config=has_pool_config)

    resolved_prompts: List[Dict[str, Any]] = []
    for idx, subset in enumerate(subsets):
        if not isinstance(subset, dict):
            raise ValueError("Each [[prompt.subset]] entry must be a TOML table/object.")

        subset_prompt = deepcopy(subset)
        subset_loras = normalize_ltx_prompt_lora_entries(subset_prompt.pop("loras", None))
        lora_mode = subset_prompt.pop("lora_mode", None)
        if lora_mode is None:
            lora_mode = "extend" if subset_loras else "extend"
        if lora_mode not in VALID_LORA_MODES:
            raise ValueError(f"Invalid lora_mode {lora_mode!r}. Expected one of {sorted(VALID_LORA_MODES)}.")

        prompt_dict = deepcopy(root_prompt)
        prompt_dict.update(subset_prompt)
        prompt_dict.pop("use_image_pool", None)
        prompt_dict["resolved_loras"] = (
            [*combined_baseline, *subset_loras] if lora_mode == "extend" else subset_loras
        )
        assigned_image = image_assignments[idx]
        if assigned_image is not None:
            prompt_dict["image_path"] = assigned_image
        else:
            prompt_dict.pop("image_path", None)
        prompt_dict["enum"] = idx
        resolved_prompts.append(prompt_dict)

    return resolved_prompts


def load_ltx_prompt_file_with_resolved_loras(
    prompt_file: str | Path, *, baseline_loras: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    prompt_path = Path(prompt_file)
    with prompt_path.open("r", encoding="utf-8") as f:
        data = toml.load(f)
    return resolve_ltx_prompt_file_data(data, baseline_loras=baseline_loras)
