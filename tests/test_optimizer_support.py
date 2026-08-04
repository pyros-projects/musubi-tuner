from pathlib import Path
from types import SimpleNamespace

import torch

from musubi_tuner.training.trainer_base import NetworkTrainer


def test_trainer_supports_vendored_prodigy_plus_schedule_free():
    trainer = NetworkTrainer()
    params = [torch.nn.Parameter(torch.ones(8))]
    args = SimpleNamespace(
        optimizer_type="ProdigyPlusScheduleFree",
        optimizer_args=["betas=(0.9, 0.99)", "weight_decay=0.0"],
        learning_rate=1.0,
        lr_scheduler="constant",
        max_grad_norm=0.0,
    )

    optimizer_name, optimizer_args, optimizer, _train_fn, _eval_fn = trainer.get_optimizer(args, params)

    repo_src = Path(__file__).resolve().parents[1] / "src"
    optimizer_module = Path(__import__(optimizer.__class__.__module__, fromlist=["__file__"]).__file__).resolve()
    assert optimizer_module.is_relative_to(repo_src)
    assert optimizer_name == "prodigyplus.prodigy_plus_schedulefree.ProdigyPlusScheduleFree"
    assert "betas=(0.9, 0.99)" in optimizer_args
    assert trainer.is_schedulefree_optimizer(optimizer, args)
    assert trainer.get_lr_scheduler(args, optimizer, 1).get_last_lr() == [1.0]

    params[0].square().mean().backward()
    optimizer.step()
    assert torch.isfinite(params[0]).all()


def test_prodigy_step_logs_include_dynamic_learning_rate_values():
    trainer = NetworkTrainer()
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    optimizer.param_groups[0].update(d=0.25, effective_lr=0.4)

    logs = trainer.generate_step_logs(
        SimpleNamespace(optimizer_type="ProdigyPlusScheduleFree"),
        current_loss=1.0,
        avr_loss=1.0,
        lr_scheduler=trainer.get_dummy_scheduler(optimizer),
        lr_descriptions=["unet"],
        optimizer=optimizer,
    )

    assert logs["lr/d/unet"] == 0.25
    assert logs["lr/effective_lr/unet"] == 0.4
