"""Qwen-Image 2.1 text encoder: the Qwen3-VL-8B text tower, text-to-image prompts only.

Mirrors diffusers' QwenImage21Pipeline / ComfyUI's ``text_encoders/qwen_image21.py``:
raw chat template with the "Comprehend and analyze the provided prompt." system turn,
the last decoder layer's output *before* the final RMSNorm, and the system turn dropped
(everything before the second ``<|im_start|>``). The vision tower and LM head are not loaded.

Weights come from a local Comfy file, bf16 or INT8 ConvRot (``qwen3vl_8b_int8_convrot``):
quantized Linears run through comfy-kitchen, the quantized embedding table dequantizes only
the looked-up rows.
"""

import logging

import torch
import torch.nn as nn
from accelerate import init_empty_weights

from musubi_tuner.modules.int8_optimization_utils import _convrot_hadamard, apply_int8_convrot_monkey_patch
from musubi_tuner.qwen_image21.qwen_image21_utils import scan_int8_convrot
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)

# The Qwen3-VL family shares one tokenizer; Krea 2 already pulls this repo's.
QWEN3_VL_TOKENIZER_REPO = "Qwen/Qwen3-VL-4B-Instruct"
SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_TEMPLATE = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
IM_START_ID = 151644


def _text_config():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    config = Qwen3VLTextConfig(
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        rope_theta=5_000_000,
        rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [24, 20, 20]},
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
        bos_token_id=151643,
        eos_token_id=151645,
    )
    config._attn_implementation = "sdpa"
    return config


class Int8ConvRotEmbedding(nn.Module):
    """Row-wise INT8 ConvRot embedding table; only the looked-up rows are dequantized and un-rotated."""

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int, output_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=torch.int8), requires_grad=False)
        self.register_buffer("weight_scale", torch.empty(num_embeddings, 1, dtype=torch.float32))
        self.group_size = group_size
        self.output_dtype = output_dtype

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows = self.weight[input_ids].float() * self.weight_scale[input_ids]
        return _convrot_hadamard(rows, self.group_size).to(self.output_dtype)


def _te_key(key: str):
    """Checkpoint key -> Qwen3VLTextModel key, or None for tensors the text tower does not use."""
    if not key.startswith("model.") or key.startswith("model.visual.") or key.endswith(".comfy_quant"):
        return None
    key = key[len("model.") :]
    if key.startswith("language_model."):
        key = key[len("language_model.") :]
    if key == "norm.weight":  # the DiT reads the last layer before the final norm
        return None
    return key


def _te_module(source: str):
    name = _te_key(source + ".weight")
    return None if name is None else name[: -len(".weight")]


def load_qwen_image21_text_encoder(path: str, device="cuda", dtype: torch.dtype = torch.bfloat16):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    device = torch.device(device)
    config = _text_config()
    with MemoryEfficientSafeOpen(path) as reader:
        int8_layers = scan_int8_convrot(reader, _te_module)
        embed_config = int8_layers.pop("embed_tokens", None)

        with init_empty_weights():
            model = Qwen3VLTextModel(config)
            model.norm = nn.Identity()
            if embed_config is not None:
                model.embed_tokens = Int8ConvRotEmbedding(config.vocab_size, config.hidden_size, embed_config.group_size, dtype)
        if int8_layers:
            apply_int8_convrot_monkey_patch(model, int8_layers)
            logger.info("Loading INT8 ConvRot Qwen3-VL text encoder (%d quantized Linears)", len(int8_layers))

        quantized = {f"{name}.weight" for name in int8_layers}
        if embed_config is not None:
            quantized.add("embed_tokens.weight")
        sd = {}
        for key in reader.keys():
            name = _te_key(key)
            if name is None:
                continue
            if name in quantized or name.endswith(".weight_scale"):
                sd[name] = reader.get_tensor(key, device=device)  # keep I8 payload / F32 scale
            else:
                sd[name] = reader.get_tensor(key, device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Qwen3-VL text encoder checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
    model.to(device)  # the rotary inv_freq buffer was built on CPU
    logger.info(f"Loaded Qwen-Image 2.1 text encoder from {path}")
    return model.eval().requires_grad_(False)


def load_tokenizer(tokenizer_repo: str = QWEN3_VL_TOKENIZER_REPO):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_repo)


@torch.no_grad()
def encode_prompts(tokenizer, text_encoder, prompts: list[str]) -> list[torch.Tensor]:
    """Return one (tokens, 4096) hidden-state tensor per prompt, system turn dropped.

    Prompts run one at a time, as in Comfy: a padded batch switches attention kernels, and the
    INT8 activation rounding amplifies that into visibly different states for the same prompt.
    """
    device = next(text_encoder.parameters()).device
    outputs = []
    for prompt in prompts:
        # Qwen has no BOS token, so an empty prompt would leave the encoder nothing to read
        input_ids = tokenizer(PROMPT_TEMPLATE.format(prompt if prompt else " "), return_tensors="pt").input_ids.to(device)
        hidden = text_encoder(input_ids=input_ids, use_cache=False).last_hidden_state[0]
        starts = (input_ids[0] == IM_START_ID).nonzero().flatten()
        outputs.append(hidden[int(starts[1]) :])
    return outputs
