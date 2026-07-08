import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musubi_tuner.qwen_image.qwen_image_model import QwenImageTransformer2DModel


class _TrackedBlock(torch.nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = index
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.apply_calls = 0

    def _apply(self, fn):
        self.apply_calls += 1
        return super()._apply(fn)


class _FakeOffloader:
    def __init__(self, blocks_to_swap: int, futures: list[int] | None = None):
        self.blocks_to_swap = blocks_to_swap
        self.forward_only = False
        self.futures = {idx: object() for idx in (futures or [])}
        self.wait_calls = []
        self.prepare_calls = []

    def _wait_blocks_move(self, block_idx: int):
        self.wait_calls.append(block_idx)
        self.futures.pop(block_idx, None)

    def prepare_block_devices_before_forward(self, blocks):
        self.prepare_calls.append((self.blocks_to_swap, len(blocks)))


def _make_qwen_shell(blocks_to_swap: int = 3, offloader: _FakeOffloader | None = None):
    model = QwenImageTransformer2DModel.__new__(QwenImageTransformer2DModel)
    torch.nn.Module.__init__(model)
    model.transformer_blocks = torch.nn.ModuleList(_TrackedBlock(i) for i in range(6))
    model.blocks_to_swap = blocks_to_swap
    model.offloader = offloader
    return model


def test_qwen_sampling_block_swap_override_zero_disables_and_restores():
    offloader = _FakeOffloader(blocks_to_swap=3, futures=[1, 4])
    model = _make_qwen_shell(blocks_to_swap=3, offloader=offloader)

    assert hasattr(model, "override_block_swap_for_sampling")
    with model.override_block_swap_for_sampling(0, torch.device("cpu")) as effective_count:
        assert effective_count == 0
        assert model.blocks_to_swap == 0
        assert offloader.blocks_to_swap == 0
        assert offloader.futures == {}
        assert all(block.apply_calls > 0 for block in model.transformer_blocks)

    assert model.blocks_to_swap == 3
    assert offloader.blocks_to_swap == 3
    assert offloader.wait_calls == [1, 4]
    assert offloader.prepare_calls[-1] == (3, 6)


def test_qwen_sampling_block_swap_override_positive_count_restores():
    offloader = _FakeOffloader(blocks_to_swap=3)
    model = _make_qwen_shell(blocks_to_swap=3, offloader=offloader)

    with model.override_block_swap_for_sampling(1, torch.device("cpu")) as effective_count:
        assert effective_count == 1
        assert model.blocks_to_swap == 1
        assert offloader.blocks_to_swap == 1
        assert offloader.prepare_calls[-1] == (1, 6)

    assert model.blocks_to_swap == 3
    assert offloader.blocks_to_swap == 3
    assert offloader.prepare_calls[-1] == (3, 6)


def test_qwen_sampling_block_swap_override_restores_after_exception():
    offloader = _FakeOffloader(blocks_to_swap=3)
    model = _make_qwen_shell(blocks_to_swap=3, offloader=offloader)

    with pytest.raises(RuntimeError, match="boom"):
        with model.override_block_swap_for_sampling(0, torch.device("cpu")):
            raise RuntimeError("boom")

    assert model.blocks_to_swap == 3
    assert offloader.blocks_to_swap == 3
    assert offloader.prepare_calls[-1] == (3, 6)


def test_qwen_sampling_block_swap_override_rejects_invalid_requests():
    model = _make_qwen_shell(blocks_to_swap=0, offloader=None)

    with pytest.raises(ValueError, match="non-negative"):
        with model.override_block_swap_for_sampling(-1, torch.device("cpu")):
            pass

    with pytest.raises(ValueError, match="Cannot swap more than 5"):
        with model.override_block_swap_for_sampling(6, torch.device("cpu")):
            pass

    with pytest.raises(ValueError, match="requires initialized block swap"):
        with model.override_block_swap_for_sampling(1, torch.device("cpu")):
            pass
