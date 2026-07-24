from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BatchObservation:
    """Small, model-independent snapshot produced after an optimizer step."""

    epoch: int
    batch: int
    steps_per_epoch: Optional[int]
    loss: float
    accuracy: float
    grad_norm: float
    proxy_metric: Optional[float] = None


@dataclass(frozen=True)
class ControllerDecision:
    """A controller action returned to the training loop for observability."""

    action: str
    old_value: float
    new_value: float
    reward: Optional[float]


class TrainingController:
    """Lifecycle contract for online hyperparameter controllers."""

    def on_batch_end(self, observation: BatchObservation) -> Optional[ControllerDecision]:
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state):
        raise NotImplementedError
