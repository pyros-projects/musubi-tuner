import torch

from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.training.trainer_base import NetworkTrainer, _get_resume_position


def test_autoresume_finds_latest_valid_state(tmp_path):
    step_state = tmp_path / "run-step00000010-state"
    step_state.mkdir()
    torch.save({"last_epoch": 10}, step_state / "scheduler.bin")

    final_state = tmp_path / "run-state"
    final_state.mkdir()
    torch.save({"last_epoch": 20}, final_state / "scheduler.bin")

    (tmp_path / "incomplete-state").mkdir()

    args = setup_parser_common().parse_args(["--autoresume"])
    args.output_dir = str(tmp_path)

    class FakeAccelerator:
        def register_save_state_pre_hook(self, _hook):
            pass

        def register_load_state_pre_hook(self, _hook):
            pass

        def load_state(self, path):
            self.loaded = path

    accelerator = FakeAccelerator()
    recovered_step = NetworkTrainer()._register_hooks_and_resume(args, accelerator, object())

    assert args.autoresume
    assert args.resume == str(final_state)
    assert accelerator.loaded == str(final_state)
    assert recovered_step == 20


def test_autoresume_position_skips_completed_batches():
    assert _get_resume_position(0, 23, 2) == (0, 0)
    assert _get_resume_position(22, 23, 2) == (0, 44)
    assert _get_resume_position(23, 23, 2) == (1, 0)
    assert _get_resume_position(28, 23, 2) == (1, 10)


def test_autoresume_prodigy_states_without_scheduler(tmp_path):
    # ProdigyPlusScheduleFree registers no LR scheduler, so accelerate state dirs
    # contain only optimizer/RNG files — autoresume must still find and rank them.
    for step in (50, 100):
        state = tmp_path / f"run-step{step:08d}-state"
        state.mkdir()
        torch.save({}, state / "optimizer.bin")
        (state / "random_states_0.pkl").write_bytes(b"x")

    args = setup_parser_common().parse_args(["--autoresume"])
    args.output_dir = str(tmp_path)

    class FakeAccelerator:
        def register_save_state_pre_hook(self, _hook):
            pass

        def register_load_state_pre_hook(self, _hook):
            pass

        def load_state(self, path):
            self.loaded = path

    accelerator = FakeAccelerator()
    recovered_step = NetworkTrainer()._register_hooks_and_resume(args, accelerator, object())

    assert args.resume == str(tmp_path / "run-step00000100-state")
    assert accelerator.loaded == str(tmp_path / "run-step00000100-state")
    assert recovered_step == 100


def test_network_delta_w_norm_matches_materialized():
    class FakeLora(torch.nn.Module):
        def __init__(self, in_f, rank, out_f, alpha):
            super().__init__()
            self.lora_down = torch.nn.Linear(in_f, rank, bias=False)
            self.lora_up = torch.nn.Linear(rank, out_f, bias=False)
            self.scale = alpha / rank

    torch.manual_seed(0)
    class FakeNetwork:
        unet_loras = [FakeLora(16, 4, 24, 2.0), FakeLora(8, 2, 8, 2.0)]
        text_encoder_loras = []

    expected = sum(
        l.scale * torch.linalg.matrix_norm(l.lora_up.weight.float() @ l.lora_down.weight.float()).item()
        for l in FakeNetwork.unet_loras
    )
    got = NetworkTrainer._network_delta_w_norm(FakeNetwork())
    assert abs(got - expected) < 1e-4, (got, expected)

    class NotLora:
        unet_loras = [object()]

    assert NetworkTrainer._network_delta_w_norm(NotLora()) is None
    assert NetworkTrainer._network_delta_w_norm(object()) is None


def test_network_delta_w_stats_groups_blocks_and_flags_hot():
    class FakeLora(torch.nn.Module):
        def __init__(self, name, gain):
            super().__init__()
            self.lora_name = name
            self.lora_down = torch.nn.Linear(4, 2, bias=False)
            self.lora_up = torch.nn.Linear(2, 4, bias=False)
            with torch.no_grad():
                self.lora_down.weight.copy_(torch.eye(2, 4))
                self.lora_up.weight.copy_(gain * torch.eye(4, 2))
            self.scale = 1.0

    # ||up @ down||_F = gain * sqrt(2) per module
    class FakeNetwork:
        unet_loras = [
            FakeLora("lora_unet_blocks_0_mlp_fc1", 1.0),
            FakeLora("lora_unet_blocks_0_mlp_fc2", 1.0),
            FakeLora("lora_unet_blocks_1_mlp_fc1", 4.0),
            FakeLora("lora_unet_token_refiner_blocks_0_attn_qkv_proj", 1.0),
        ]
        text_encoder_loras = []

    stats = NetworkTrainer._network_delta_w_stats(FakeNetwork())

    root2 = 2**0.5
    assert abs(stats["total"] - 7 * root2) < 1e-4
    # buckets: b0 = 2*sqrt(2), b1 = 4*sqrt(2), r0 = sqrt(2) -> median b0, hot b1
    assert abs(stats["block_median"] - 2 * root2) < 1e-4
    assert stats["hot_block"] == "b1"
    assert abs(stats["hot_ratio"] - 2.0) < 1e-4
