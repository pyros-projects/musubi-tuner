from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml


VALID_LORA_MODES = {"extend", "replace"}


def validate_prompt_lora_entry(entry: Any) -> Dict[str, Any]:
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


def normalize_prompt_lora_entries(entries: Any) -> List[Dict[str, Any]]:
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError("Prompt-file 'loras' must be a list of TOML tables.")
    return [validate_prompt_lora_entry(entry) for entry in entries]


def resolve_prompt_file_data(
    data: Dict[str, Any], *, baseline_loras: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    prompt_root = data.get("prompt")
    if not isinstance(prompt_root, dict):
        raise ValueError("Prompt TOML must contain a [prompt] table.")

    subsets = prompt_root.get("subset")
    if not isinstance(subsets, list) or not subsets:
        raise ValueError("Prompt TOML must contain at least one [[prompt.subset]] entry.")

    root_prompt = deepcopy(prompt_root)
    root_prompt.pop("subset", None)
    root_baseline_loras = normalize_prompt_lora_entries(root_prompt.pop("loras", None))
    cli_baseline_loras = normalize_prompt_lora_entries(baseline_loras)
    combined_baseline = [*cli_baseline_loras, *root_baseline_loras]

    resolved_prompts: List[Dict[str, Any]] = []
    for idx, subset in enumerate(subsets):
        if not isinstance(subset, dict):
            raise ValueError("Each [[prompt.subset]] entry must be a TOML table/object.")

        subset_prompt = deepcopy(subset)
        subset_loras = normalize_prompt_lora_entries(subset_prompt.pop("loras", None))
        lora_mode = subset_prompt.pop("lora_mode", None)
        if lora_mode is None:
            lora_mode = "extend"
        if lora_mode not in VALID_LORA_MODES:
            raise ValueError(f"Invalid lora_mode {lora_mode!r}. Expected one of {sorted(VALID_LORA_MODES)}.")

        prompt_dict = deepcopy(root_prompt)
        prompt_dict.update(subset_prompt)
        prompt_dict["resolved_loras"] = (
            [*combined_baseline, *subset_loras] if lora_mode == "extend" else subset_loras
        )
        prompt_dict["enum"] = idx
        resolved_prompts.append(prompt_dict)

    return resolved_prompts


def load_prompt_file_with_resolved_loras(
    prompt_file: str | Path, *, baseline_loras: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    prompt_path = Path(prompt_file)
    with prompt_path.open("r", encoding="utf-8") as f:
        data = toml.load(f)
    return resolve_prompt_file_data(data, baseline_loras=baseline_loras)
