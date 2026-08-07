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
