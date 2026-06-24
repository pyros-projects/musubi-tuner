import sys
import tempfile
import types
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musubi_tuner.krea2_train_network import Krea2NetworkTrainer


class _FakePrintAccelerator:
    def __init__(self):
        self.messages = []

    def print(self, message):
        self.messages.append(str(message))


def _sample_krea2_lora_state():
    return {
        "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.ones(2, 4),
        "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.ones(4, 2),
        "lora_unet_blocks_0_attn_wq.alpha": torch.tensor(2.0),
        "lora_unet_txtfusion_layerwise_blocks_0_attn_wq.lora_down.weight": torch.ones(2, 4) * 2,
        "lora_unet_txtfusion_layerwise_blocks_0_attn_wq.lora_up.weight": torch.ones(4, 2) * 3,
        "lora_unet_txtfusion_layerwise_blocks_0_attn_wq.alpha": torch.tensor(2.0),
        "lora_unet_txtfusion_refiner_blocks_1_mlp_down.lora_down.weight": torch.ones(2, 4) * 4,
        "lora_unet_txtfusion_refiner_blocks_1_mlp_down.lora_up.weight": torch.ones(4, 2) * 5,
        "lora_unet_txtfusion_refiner_blocks_1_mlp_down.alpha": torch.tensor(2.0),
        "lora_unet_txtfusion_projector.lora_down.weight": torch.ones(2, 4) * 6,
        "lora_unet_txtfusion_projector.lora_up.weight": torch.ones(4, 2) * 7,
        "lora_unet_txtfusion_projector.alpha": torch.tensor(2.0),
        "lora_unet_tmlp_0.lora_down.weight": torch.ones(2, 4) * 8,
        "lora_unet_tmlp_0.lora_up.weight": torch.ones(4, 2) * 9,
        "lora_unet_tmlp_0.alpha": torch.tensor(2.0),
        "lora_unet_txtmlp_1.lora_down.weight": torch.ones(2, 4) * 10,
        "lora_unet_txtmlp_1.lora_up.weight": torch.ones(4, 2) * 11,
        "lora_unet_txtmlp_1.alpha": torch.tensor(2.0),
        "lora_unet_tproj_1.lora_down.weight": torch.ones(2, 4) * 12,
        "lora_unet_tproj_1.lora_up.weight": torch.ones(4, 2) * 13,
        "lora_unet_tproj_1.alpha": torch.tensor(2.0),
        "lora_unet_first.lora_down.weight": torch.ones(2, 4) * 14,
        "lora_unet_first.lora_up.weight": torch.ones(4, 2) * 15,
        "lora_unet_first.alpha": torch.tensor(2.0),
        "lora_unet_last_linear.lora_down.weight": torch.ones(2, 4) * 16,
        "lora_unet_last_linear.lora_up.weight": torch.ones(4, 2) * 17,
        "lora_unet_last_linear.alpha": torch.tensor(2.0),
    }


def test_krea2_post_save_hook_writes_native_comfy_checkpoint():
    trainer = Krea2NetworkTrainer()
    accelerator = _FakePrintAccelerator()
    output_dir = tempfile.mkdtemp()
    ckpt_file = Path(output_dir) / "test.safetensors"
    save_file(_sample_krea2_lora_state(), str(ckpt_file), metadata={"ss_output_name": "test-krea2"})
    args = types.SimpleNamespace(
        convert_to_comfy=True,
        save_original_lora=True,
        output_dir=output_dir,
        huggingface_repo_id=None,
    )

    trainer.post_save_checkpoint_hook(args, str(ckpt_file), ckpt_file.name, accelerator)

    comfy_file = ckpt_file.with_name("test.comfy.safetensors")
    assert comfy_file.exists()
    assert ckpt_file.exists()

    comfy_sd = load_file(str(comfy_file))
    assert "diffusion_model.blocks.0.attn.wq.lora_down.weight" in comfy_sd
    assert "diffusion_model.blocks.0.attn.wq.lora_up.weight" in comfy_sd
    assert "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lora_up.weight" in comfy_sd
    assert "diffusion_model.txtfusion.refiner_blocks.1.mlp.down.lora_down.weight" in comfy_sd
    assert "diffusion_model.txtfusion.projector.lora_up.weight" in comfy_sd
    assert "diffusion_model.tmlp.0.lora_down.weight" in comfy_sd
    assert "diffusion_model.txtmlp.1.lora_up.weight" in comfy_sd
    assert "diffusion_model.tproj.1.lora_down.weight" in comfy_sd
    assert "diffusion_model.first.lora_up.weight" in comfy_sd
    assert "diffusion_model.last.linear.lora_down.weight" in comfy_sd
    assert not any(".alpha" in key for key in comfy_sd)
    assert not any("txtfusion.layerwise.blocks" in key for key in comfy_sd)
    assert not any("txtfusion.refiner.blocks" in key for key in comfy_sd)

    with safe_open(str(comfy_file), framework="pt") as f:
        assert f.metadata()["ss_output_name"] == "test-krea2"


def test_krea2_converter_folds_non_default_alpha_into_up_weight():
    from musubi_tuner.krea2.convert_lora_to_comfy import convert_state_dict_to_comfy

    converted = convert_state_dict_to_comfy(
        {
            "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.ones(2, 4),
            "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.ones(4, 2),
            "lora_unet_blocks_0_attn_wq.alpha": torch.tensor(1.0),
            "lora_unet_last_linear.lora_down.weight": torch.ones(2, 4),
            "lora_unet_last_linear.lora_up.weight": torch.ones(4, 2) * 3,
            "lora_unet_last_linear.alpha": torch.tensor(2.0),
        }
    )

    assert torch.allclose(converted["diffusion_model.blocks.0.attn.wq.lora_down.weight"], torch.ones(2, 4))
    assert torch.allclose(converted["diffusion_model.blocks.0.attn.wq.lora_up.weight"], torch.ones(4, 2) * 0.5)
    assert torch.allclose(converted["diffusion_model.last.linear.lora_up.weight"], torch.ones(4, 2) * 3)
    assert not any(".alpha" in key for key in converted)


def test_krea2_post_save_hook_respects_conversion_flags():
    trainer = Krea2NetworkTrainer()
    accelerator = _FakePrintAccelerator()
    output_dir = tempfile.mkdtemp()

    disabled_file = Path(output_dir) / "disabled.safetensors"
    save_file(_sample_krea2_lora_state(), str(disabled_file))
    disabled_args = types.SimpleNamespace(
        convert_to_comfy=False,
        save_original_lora=True,
        output_dir=output_dir,
        huggingface_repo_id=None,
    )
    trainer.post_save_checkpoint_hook(disabled_args, str(disabled_file), disabled_file.name, accelerator)
    assert not disabled_file.with_name("disabled.comfy.safetensors").exists()
    assert disabled_file.exists()

    remove_file = Path(output_dir) / "remove.safetensors"
    save_file(_sample_krea2_lora_state(), str(remove_file))
    remove_args = types.SimpleNamespace(
        convert_to_comfy=True,
        save_original_lora=False,
        output_dir=output_dir,
        huggingface_repo_id=None,
    )
    trainer.post_save_checkpoint_hook(remove_args, str(remove_file), remove_file.name, accelerator)
    assert remove_file.with_name("remove.comfy.safetensors").exists()
    assert not remove_file.exists()
