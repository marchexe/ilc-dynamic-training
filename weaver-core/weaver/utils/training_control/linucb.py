import json
import math
from pathlib import Path

import numpy as np

from .base import ControllerDecision, TrainingController
from weaver.utils.logger import _logger


class LinUCBLearningRateController(TrainingController):
    """Online contextual bandit that selects a multiplicative LR action.

    The reward is the relative improvement in exponentially-smoothed training
    loss over the decision window.  This is deliberately a small first
    controller: the observation/reward interface can later receive proxy
    validation metrics without coupling the bandit to a specific model.
    """

    _FEATURE_DIM = 6

    def __init__(
        self,
        optimizer,
        *,
        interval_steps=100,
        warmup_steps=200,
        actions=(0.8, 1.0, 1.2),
        min_lr=1e-6,
        max_lr=1e-2,
        ema_beta=0.95,
        alpha=0.5,
        ridge=1.0,
        seed=12345,
        log_path=None,
        observe_only=False,
    ):
        if interval_steps < 1:
            raise ValueError("interval_steps must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0 <= ema_beta < 1:
            raise ValueError("ema_beta must be in [0, 1)")
        if min_lr <= 0 or max_lr <= min_lr:
            raise ValueError("expected 0 < min_lr < max_lr")
        if ridge <= 0:
            raise ValueError("ridge must be positive")

        actions = tuple(float(value) for value in actions)
        if not actions or any(value <= 0 for value in actions):
            raise ValueError("actions must contain positive LR multipliers")

        self.optimizer = optimizer
        self.interval_steps = int(interval_steps)
        self.warmup_steps = int(warmup_steps)
        self.actions = actions
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.ema_beta = float(ema_beta)
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.observe_only = bool(observe_only)
        self.log_path = None if log_path is None else Path(log_path)
        self.rng = np.random.default_rng(seed)

        n_actions = len(actions)
        eye = np.eye(self._FEATURE_DIM, dtype=np.float64) * self.ridge
        self._a = np.repeat(eye[None, :, :], n_actions, axis=0)
        self._b = np.zeros((n_actions, self._FEATURE_DIM), dtype=np.float64)
        self._action_counts = np.zeros(n_actions, dtype=np.int64)
        self._global_step = 0
        self._loss_ema = None
        self._previous_decision_loss = None
        self._previous_context = None
        self._previous_action = None

        _logger.info(
            "[training-control] initialized: controller=linucb_lr mode=%s "
            "warmup_steps=%d interval_steps=%d actions=%s lr_bounds=[%.2e, %.2e]",
            "observe" if self.observe_only else "active",
            self.warmup_steps,
            self.interval_steps,
            self.actions,
            self.min_lr,
            self.max_lr,
        )

    @staticmethod
    def _as_float(value):
        return float(value.detach().item()) if hasattr(value, "detach") else float(value)

    def _current_lr(self):
        return self._as_float(self.optimizer.param_groups[0]["lr"])

    def _set_lr(self, value):
        old_reference_lr = self._current_lr()
        effective_factor = value / old_reference_lr
        for group in self.optimizer.param_groups:
            current = group["lr"]
            group_value = self._as_float(current) * effective_factor
            if hasattr(current, "fill_"):
                current.fill_(group_value)
            else:
                group["lr"] = group_value

    def _context(self, observation):
        progress = 0.0
        if observation.steps_per_epoch:
            progress = observation.batch / observation.steps_per_epoch
        previous = self._previous_decision_loss
        slope = 0.0 if previous is None else (previous - self._loss_ema) / max(abs(previous), 1e-12)
        lr_position = math.log(self._current_lr() / self.min_lr) / math.log(self.max_lr / self.min_lr)
        return np.asarray(
            [
                1.0,
                float(np.clip(progress, 0.0, 1.0)),
                math.log(max(self._loss_ema, 1e-12)),
                float(np.clip(slope, -1.0, 1.0)),
                math.log1p(max(observation.grad_norm, 0.0)),
                float(np.clip(lr_position, 0.0, 1.0)),
            ],
            dtype=np.float64,
        )

    def _update_previous_action(self):
        if self._previous_action is None:
            return None
        reward = (self._previous_decision_loss - self._loss_ema) / max(
            abs(self._previous_decision_loss), 1e-12
        )
        reward = float(np.clip(reward, -1.0, 1.0))
        idx = self._previous_action
        context = self._previous_context
        self._a[idx] += np.outer(context, context)
        self._b[idx] += reward * context
        return reward

    def _select_action(self, context):
        unseen = np.flatnonzero(self._action_counts == 0)
        if unseen.size:
            return int(self.rng.choice(unseen))

        scores = []
        for idx in range(len(self.actions)):
            a_inv = np.linalg.inv(self._a[idx])
            theta = a_inv @ self._b[idx]
            uncertainty = math.sqrt(max(float(context @ a_inv @ context), 0.0))
            scores.append(float(theta @ context + self.alpha * uncertainty))
        best = np.flatnonzero(np.isclose(scores, np.max(scores)))
        return int(self.rng.choice(best))

    def _write_event(self, payload):
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def on_batch_end(self, observation):
        self._global_step += 1
        if self._loss_ema is None:
            self._loss_ema = float(observation.loss)
        else:
            self._loss_ema = self.ema_beta * self._loss_ema + (1 - self.ema_beta) * observation.loss

        if self._global_step < self.warmup_steps:
            return None
        if (self._global_step - self.warmup_steps) % self.interval_steps:
            return None

        reward = None if self.observe_only else self._update_previous_action()
        context = self._context(observation)
        action_idx = self._select_action(context)
        factor = self.actions[action_idx]
        old_lr = self._current_lr()
        proposed_lr = float(np.clip(old_lr * factor, self.min_lr, self.max_lr))
        new_lr = old_lr if self.observe_only else proposed_lr
        if not self.observe_only:
            self._set_lr(new_lr)

        self._action_counts[action_idx] += 1
        self._previous_action = None if self.observe_only else action_idx
        self._previous_context = None if self.observe_only else context
        self._previous_decision_loss = self._loss_ema

        action = f"observe_lr_x{factor:g}" if self.observe_only else f"lr_x{factor:g}"
        self._write_event(
            {
                "action": action,
                "action_counts": self._action_counts.tolist(),
                "action_factor": factor,
                "accuracy": observation.accuracy,
                "batch": observation.batch,
                "context": context.tolist(),
                "controller": "linucb_lr",
                "epoch": observation.epoch,
                "global_step": self._global_step,
                "grad_norm": observation.grad_norm,
                "loss": observation.loss,
                "loss_ema": self._loss_ema,
                "new_lr": new_lr,
                "old_lr": old_lr,
                "proposed_lr": proposed_lr,
                "reward": reward,
                "schema_version": 1,
                "steps_per_epoch": observation.steps_per_epoch,
            }
        )
        reward_text = "n/a" if reward is None else f"{reward:+.6f}"
        _logger.info(
            "[training-control] decision: step=%d epoch=%d batch=%d mode=%s "
            "action=%s old_lr=%.3e proposed_lr=%.3e applied_lr=%.3e "
            "reward=%s loss_ema=%.6f",
            self._global_step,
            observation.epoch,
            observation.batch,
            "observe" if self.observe_only else "active",
            action,
            old_lr,
            proposed_lr,
            new_lr,
            reward_text,
            self._loss_ema,
        )
        return ControllerDecision(action=action, old_value=old_lr, new_value=new_lr, reward=reward)

    def state_dict(self):
        return {
            "a": self._a,
            "b": self._b,
            "action_counts": self._action_counts,
            "global_step": self._global_step,
            "loss_ema": self._loss_ema,
            "previous_decision_loss": self._previous_decision_loss,
            "previous_context": self._previous_context,
            "previous_action": self._previous_action,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state):
        self._a = np.asarray(state["a"], dtype=np.float64)
        self._b = np.asarray(state["b"], dtype=np.float64)
        self._action_counts = np.asarray(state["action_counts"], dtype=np.int64)
        self._global_step = int(state["global_step"])
        self._loss_ema = state["loss_ema"]
        self._previous_decision_loss = state["previous_decision_loss"]
        self._previous_context = state["previous_context"]
        self._previous_action = state["previous_action"]
        self.rng.bit_generator.state = state["rng_state"]
        _logger.info(
            "[training-control] restored state: global_step=%d action_counts=%s",
            self._global_step,
            self._action_counts.tolist(),
        )
