#!/usr/bin/env python3
"""Repack LTX-2.5 split release files into the 2.3-style single-file checkpoint.

LTX-2.5 ships the transformer, video VAE, audio VAE (+vocoder), and text encoder
as separate safetensors files. The LTX-2 trainer expects the 2.3 layout: one file
holding ``model.diffusion_model.*`` + ``vae.*`` + ``audio_vae.*`` + ``vocoder.*``
+ ``text_embedding_projection.*`` with a merged ``config`` JSON in its metadata.
This tool reassembles that layout so ``--ltx2_checkpoint`` works unchanged.

Tensors are copied as raw byte ranges (no torch materialization), so peak RSS
stays at the copy-buffer size regardless of checkpoint size.

Usage:
    python ltx2_repack.py \\
        --transformer path/to/ltx-2.5-22b-dev-transformer-bf16.safetensors \\
        --video_vae path/to/ltx-2.5-video-vae-conv-bf16.safetensors \\
        --audio_vae path/to/ltx-2.5-audio-vae-bf16.safetensors \\
        --text_encoder path/to/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \\
        --output path/to/ltx-2.5-22b-dev.safetensors

Notes:
    - Use the ``-conv`` video VAE (2.3-architecture decoder). The default 2.5
      video VAE has a new diffusion decoder the trainer does not support yet;
      its config is rejected here.
    - Only the four ``text_embedding_projection.*`` tensors are taken from the
      text encoder file; the Gemma-4 weights themselves stay in their own file.
    - The ``--transformer`` slot accepts either the bf16 release file or the
      official ``comfy-int8-convrot`` variant (I8 weights + F32 per-channel
      scales + ``.comfy_quant`` markers are passed through verbatim). Other
      quantized formats (nvfp4, ...) are rejected; the VAE and text encoder
      slots remain bf16-only.
    - The license text and provenance metadata of the transformer file are
      carried through to the output.
"""

import argparse
import json
import logging
import os
import struct

logger = logging.getLogger(__name__)

_COPY_CHUNK = 64 * 1024 * 1024
_ALLOWED_DTYPES = {"BF16", "F32", "F16", "F64"}
_QUANT_KEY_MARKERS = (".comfy_quant", ".weight_scale", ".weight_scale_2", ".absmax", ".quant_map")
# Key markers that are NOT part of the int8-convrot layout (nvfp4 & friends).
_INT8_FORBIDDEN_KEY_MARKERS = (".weight_scale_2", ".absmax", ".quant_map")
_INT8_ALLOWED_DTYPES = _ALLOWED_DTYPES | {"I8", "U8"}
_TEXT_PROJECTION_PREFIX = "text_embedding_projection."


def _read_safetensors_header(path: str) -> tuple[dict, dict, int]:
    """Return (metadata, tensor_entries, data_start) without reading tensor data."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    metadata = header.pop("__metadata__", {})
    return metadata, header, 8 + header_size


def _config_from_metadata(metadata: dict, path: str) -> dict:
    raw = metadata.get("config")
    if raw is None:
        raise ValueError(f"{path}: no 'config' entry in safetensors metadata — not an LTX release file")
    return json.loads(raw)


def _check_dtypes(tensors: dict, path: str, keys=None) -> None:
    for key in keys if keys is not None else tensors:
        if any(marker in key for marker in _QUANT_KEY_MARKERS):
            raise ValueError(
                f"{path}: quantization key '{key}' found — this looks like an int8/nvfp4 variant. "
                "Repack requires the bf16 release files."
            )
        dtype = tensors[key]["dtype"]
        if dtype not in _ALLOWED_DTYPES:
            raise ValueError(
                f"{path}: tensor '{key}' has dtype {dtype} — this looks like a quantized variant. "
                "Repack requires the bf16 release files."
            )


class _Section:
    """One input file contributing tensors (with an output key prefix) and config sections."""

    def __init__(self, name: str, path: str, key_map: dict[str, str], config: dict):
        self.name = name
        self.path = path
        self.key_map = key_map  # source key -> output key
        self.config = config  # config sections contributed by this file


def _validate_int8_convrot_marker(path: str, tensors: dict, data_start: int, marker_keys: list[str]) -> None:
    """Parse one ``.comfy_quant`` marker (U8 JSON bytes) and require int8-convrot."""
    info = tensors[sorted(marker_keys)[0]]
    begin, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + begin)
        raw = f.read(end - begin)
    try:
        marker = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: unreadable .comfy_quant marker ({error}) — unsupported quantized variant") from error
    params = marker.get("params", {})
    fmt = marker.get("format")
    convrot = marker.get("convrot", params.get("convrot", False))
    if fmt != "int8_tensorwise" or not convrot:
        raise ValueError(
            f"{path}: quantized transformer has format {fmt!r} (convrot={convrot}) — "
            "only int8-convrot is supported here (nvfp4 and other variants are not)."
        )


def _check_dtypes_int8_transformer(tensors: dict, path: str) -> None:
    for key in tensors:
        if any(marker in key for marker in _INT8_FORBIDDEN_KEY_MARKERS):
            raise ValueError(
                f"{path}: quantization key '{key}' is not part of the int8-convrot layout — unsupported variant."
            )
        dtype = tensors[key]["dtype"]
        if dtype not in _INT8_ALLOWED_DTYPES:
            raise ValueError(
                f"{path}: tensor '{key}' has dtype {dtype} — unsupported quantized variant "
                "(only the bf16 or int8-convrot transformer files are accepted)."
            )


def _load_transformer_section(path: str) -> tuple[_Section, dict, int]:
    """Returns (section, metadata, int8_linear_count) — count is 0 for the bf16 file."""
    metadata, tensors, data_start = _read_safetensors_header(path)
    bad = [k for k in tensors if not k.startswith("model.diffusion_model.")]
    if bad or not tensors:
        raise ValueError(
            f"{path}: expected only 'model.diffusion_model.*' keys (found e.g. {bad[:3] or 'no tensors'}) — "
            "is this really the 2.5 transformer file?"
        )
    config = _config_from_metadata(metadata, path)
    if "transformer" not in config:
        raise ValueError(f"{path}: metadata config has no 'transformer' section")
    marker_keys = [k for k in tensors if k.endswith(".comfy_quant")]
    if marker_keys:
        _validate_int8_convrot_marker(path, tensors, data_start, marker_keys)
        _check_dtypes_int8_transformer(tensors, path)
    else:
        _check_dtypes(tensors, path)
    return _Section("transformer", path, {k: k for k in tensors}, config), metadata, len(marker_keys)


def _load_video_vae_section(path: str) -> _Section:
    metadata, tensors, _ = _read_safetensors_header(path)
    if any(k.startswith("model.diffusion_model.") for k in tensors):
        raise ValueError(f"{path}: contains transformer keys — wrong file passed to --video_vae?")
    if not any(k.startswith("encoder.") for k in tensors) or not any(k.startswith("decoder.") for k in tensors):
        raise ValueError(f"{path}: expected unprefixed 'encoder.*'/'decoder.*' keys — is this the 2.5 video VAE file?")
    config = _config_from_metadata(metadata, path)
    vae_config = config.get("vae")
    if vae_config is None:
        raise ValueError(f"{path}: metadata config has no 'vae' section")
    if "encoder_blocks" not in vae_config:
        raise ValueError(
            f"{path}: 'vae' config has no 'encoder_blocks' — this looks like the new diffusion-decoder VAE. "
            "Use the '-conv' video VAE file, whose architecture matches the trainer."
        )
    _check_dtypes(tensors, path)
    return _Section("video_vae", path, {k: f"vae.{k}" for k in tensors}, {"vae": vae_config})


def _load_audio_vae_section(path: str) -> _Section:
    metadata, tensors, _ = _read_safetensors_header(path)
    bad = [k for k in tensors if not k.startswith(("audio_vae.", "vocoder."))]
    if bad or not tensors:
        raise ValueError(
            f"{path}: expected 'audio_vae.*'/'vocoder.*' keys (found e.g. {bad[:3] or 'no tensors'}) — "
            "is this really the 2.5 audio VAE file?"
        )
    config = _config_from_metadata(metadata, path)
    sections = {name: config[name] for name in ("audio_vae", "vocoder") if name in config}
    if "audio_vae" not in sections:
        raise ValueError(f"{path}: metadata config has no 'audio_vae' section")
    _check_dtypes(tensors, path)
    return _Section("audio_vae", path, {k: k for k in tensors}, sections)


def _load_text_projection_section(path: str) -> _Section:
    _, tensors, _ = _read_safetensors_header(path)
    projection_keys = [k for k in tensors if k.startswith(_TEXT_PROJECTION_PREFIX)]
    if not projection_keys:
        raise ValueError(
            f"{path}: no '{_TEXT_PROJECTION_PREFIX}*' tensors found — "
            "is this really the gemma4 'with-proj' text encoder file?"
        )
    _check_dtypes(tensors, path, keys=projection_keys)
    return _Section("text_projection", path, {k: k for k in sorted(projection_keys)}, {})


def repack(
    transformer: str,
    video_vae: str,
    audio_vae: str,
    text_encoder: str,
    output: str,
    overwrite: bool = False,
) -> dict[str, int]:
    """Merge the 2.5 split files into a 2.3-layout single file. Returns per-section tensor counts."""
    if os.path.exists(output) and not overwrite:
        raise ValueError(f"output already exists: {output} (use --overwrite to replace)")

    transformer_section, transformer_metadata, int8_linear_count = _load_transformer_section(transformer)
    if int8_linear_count:
        logger.info("Transformer input is int8-convrot (%d quantized Linears) — passing through verbatim", int8_linear_count)
    sections = [
        transformer_section,
        _load_video_vae_section(video_vae),
        _load_audio_vae_section(audio_vae),
        _load_text_projection_section(text_encoder),
    ]

    merged_config = dict(transformer_section.config)
    for section in sections[1:]:
        for name, value in section.config.items():
            if name in ("transformer", "scheduler"):
                continue
            if name in merged_config and merged_config[name] != value:
                logger.warning("config section '%s' from %s overrides transformer file's copy", name, section.path)
            merged_config[name] = value

    output_metadata = {"config": json.dumps(merged_config)}
    for carry in ("model_version", "license", "gemma_source_checkpoint", "format"):
        if carry in transformer_metadata:
            output_metadata[carry] = transformer_metadata[carry]

    # Build the output header with sequentially recomputed data offsets.
    entries = []  # (output_key, source_path, source_abs_start, num_bytes, tensor_info)
    output_tensors: dict[str, dict] = {}
    cursor = 0
    for section in sections:
        _, tensors, data_start = _read_safetensors_header(section.path)
        for source_key, output_key in section.key_map.items():
            if output_key in output_tensors:
                raise ValueError(f"duplicate output key '{output_key}' from {section.path}")
            info = tensors[source_key]
            begin, end = info["data_offsets"]
            num_bytes = end - begin
            output_tensors[output_key] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "data_offsets": [cursor, cursor + num_bytes],
            }
            entries.append((output_key, section.path, data_start + begin, num_bytes))
            cursor += num_bytes

    header = {"__metadata__": output_metadata, **output_tensors}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * padding

    total = cursor
    logger.info("Writing %s (%d tensors, %.2f GB of tensor data)...", output, len(entries), total / 1e9)
    with open(output, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        source_handles: dict[str, object] = {}
        try:
            for i, (_, source_path, start, num_bytes) in enumerate(entries):
                handle = source_handles.get(source_path)
                if handle is None:
                    handle = source_handles[source_path] = open(source_path, "rb")
                handle.seek(start)
                remaining = num_bytes
                while remaining > 0:
                    chunk = handle.read(min(_COPY_CHUNK, remaining))
                    if not chunk:
                        raise ValueError(f"unexpected EOF while copying from {source_path}")
                    out.write(chunk)
                    remaining -= len(chunk)
                if (i + 1) % 500 == 0:
                    logger.info("  %d/%d tensors copied", i + 1, len(entries))
        finally:
            for handle in source_handles.values():
                handle.close()

    counts = {}
    for section in sections:
        counts[section.name] = len(section.key_map)
    logger.info("Done. Sections: %s", counts)
    if int8_linear_count:
        logger.info(
            "Transformer section: int8-convrot (%d tensors, %d quantized Linears)",
            counts["transformer"], int8_linear_count,
        )
    logger.info("Config sections: %s", sorted(merged_config))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--transformer", required=True, help="2.5 transformer safetensors (bf16 or comfy-int8-convrot)")
    parser.add_argument("--video_vae", required=True, help="2.5 video VAE '-conv' bf16 safetensors")
    parser.add_argument("--audio_vae", required=True, help="2.5 audio VAE bf16 safetensors")
    parser.add_argument("--text_encoder", required=True, help="gemma4 'with-proj' text encoder safetensors")
    parser.add_argument("--output", required=True, help="output single-file checkpoint path")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        repack(
            transformer=args.transformer,
            video_vae=args.video_vae,
            audio_vae=args.audio_vae,
            text_encoder=args.text_encoder,
            output=args.output,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
