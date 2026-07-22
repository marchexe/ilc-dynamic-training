from types import SimpleNamespace

import numpy as np

from weaver.utils.training_control.base import BatchObservation
from weaver.utils.training_control.factory import build_training_controller
from weaver.utils.training_control.linucb import LinUCBLearningRateController


def _observation(batch, loss):
    return BatchObservation(
        epoch=0,
        batch=batch,
        steps_per_epoch=10,
        loss=loss,
        accuracy=0.5,
        grad_norm=1.0,
    )


def test_linucb_changes_lr_and_learns_reward(tmp_path):
    optimizer = SimpleNamespace(param_groups=[{"lr": 1e-3}])
    controller = LinUCBLearningRateController(
        optimizer,
        interval_steps=1,
        warmup_steps=1,
        actions=(0.5,),
        min_lr=1e-5,
        max_lr=1e-2,
        ema_beta=0,
        log_path=tmp_path / "events.jsonl",
    )

    first = controller.on_batch_end(_observation(1, 1.0))
    second = controller.on_batch_end(_observation(2, 0.8))

    assert first.action == "lr_x0.5"
    assert np.isclose(second.reward, 0.2)
    assert optimizer.param_groups[0]["lr"] == 2.5e-4
    assert controller.state_dict()["action_counts"].tolist() == [2]
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2


def test_linucb_state_round_trip():
    optimizer = SimpleNamespace(param_groups=[{"lr": 1e-3}])
    controller = LinUCBLearningRateController(
        optimizer, interval_steps=1, warmup_steps=1, actions=(0.8, 1.0), seed=7
    )
    controller.on_batch_end(_observation(1, 1.0))
    state = controller.state_dict()

    restored = LinUCBLearningRateController(
        optimizer, interval_steps=1, warmup_steps=1, actions=(0.8, 1.0), seed=99
    )
    restored.load_state_dict(state)

    np.testing.assert_allclose(restored.state_dict()["a"], state["a"])
    np.testing.assert_allclose(restored.state_dict()["b"], state["b"])
    assert restored.state_dict()["global_step"] == 1


def test_linucb_preserves_parameter_group_lr_ratio():
    optimizer = SimpleNamespace(param_groups=[{"lr": 1e-3}, {"lr": 2e-3}])
    controller = LinUCBLearningRateController(
        optimizer, interval_steps=1, warmup_steps=1, actions=(0.5,), min_lr=1e-5, max_lr=1e-2
    )

    controller.on_batch_end(_observation(1, 1.0))

    assert np.isclose(optimizer.param_groups[0]["lr"], 5e-4)
    assert np.isclose(optimizer.param_groups[1]["lr"], 1e-3)


def test_observe_only_does_not_attribute_reward_to_unapplied_action():
    optimizer = SimpleNamespace(param_groups=[{"lr": 1e-3}])
    controller = LinUCBLearningRateController(
        optimizer, interval_steps=1, warmup_steps=1, actions=(0.5,), observe_only=True, ema_beta=0
    )

    first = controller.on_batch_end(_observation(1, 1.0))
    second = controller.on_batch_end(_observation(2, 0.8))

    assert first.action == "observe_lr_x0.5"
    assert second.reward is None
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert np.all(controller.state_dict()["b"] == 0)


def test_controller_intervals_can_be_epoch_fractions(tmp_path):
    config = tmp_path / "controller.yaml"
    config.write_text(
        "type: linucb_lr\n"
        "interval_fraction: 0.05\n"
        "warmup_fraction: 0.10\n"
        "actions: [0.9, 1.0, 1.1]\n"
    )
    optimizer = SimpleNamespace(param_groups=[{"lr": 1e-3}])

    controller = build_training_controller(config, optimizer, steps_per_epoch=600)

    assert controller.interval_steps == 30
    assert controller.warmup_steps == 60
