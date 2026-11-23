"""The pretrained ParT tagger trained on the mean class probabilities of a frozen teacher ensemble (distillation).

Network options (Weaver --network-option):
  teacher_paths  list of state-dict files of pretrained_sgv_particle_transformer models (the ensemble)
  hard_weight    weight of the usual cross-entropy on the true labels; the soft term gets 1 - hard_weight

In training mode every batch also goes through the teachers (eval mode, no gradient), and the loss is
    hard_weight * CE(student, label) + (1 - hard_weight) * CE(student, mean_k softmax(teacher_k)).
The teachers are held outside the module tree, so they are not in state_dict(), parameters() or the optimizer:
a checkpoint of this network is a plain checkpoint of pretrained_sgv_particle_transformer, and evaluation or the
holdout can use either network file. Without teacher_paths (test mode) the network is the plain student.
"""
import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "pretrained_sgv_particle_transformer", Path(__file__).with_name("pretrained_sgv_particle_transformer.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

# The loss reads the teacher probabilities of the current batch from here; the model writes them in forward.
_BATCH = {}


class Distilled(torch.nn.Module):
    def __init__(self, student, teachers):
        super().__init__()
        self.student = student
        self._teachers = list(teachers)  # a plain list: not registered as submodules
        for teacher in self._teachers:
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

    def no_weight_decay(self):
        return {"student." + name for name in self.student.no_weight_decay()}

    def forward(self, *args):
        logits = self.student(*args)
        if self.training and self._teachers:
            device = logits.device
            with torch.no_grad():
                probs = []
                for teacher in self._teachers:
                    if next(teacher.parameters()).device != device:
                        teacher.to(device)
                    teacher.eval()
                    probs.append(torch.softmax(teacher(*args).float(), dim=1))
                _BATCH["teacher_probs"] = torch.stack(probs).mean(0)
        else:
            _BATCH.pop("teacher_probs", None)
        return logits

    # Checkpoints hold the student only, under the names of the plain network.
    def state_dict(self, *args, **kwargs):
        return self.student.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        return self.student.load_state_dict(state_dict, strict=strict, **kwargs)


class DistillLoss(torch.nn.Module):
    def __init__(self, hard_weight):
        super().__init__()
        if not 0.0 <= hard_weight <= 1.0:
            raise ValueError("hard_weight must be in [0, 1]")
        self.hard_weight = float(hard_weight)
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits, label):
        hard = self.ce(logits, label)
        teacher = _BATCH.pop("teacher_probs", None)
        if teacher is None:  # validation, or no teachers
            return hard
        if teacher.shape != logits.shape:
            raise RuntimeError(f"teacher probabilities {tuple(teacher.shape)} do not match logits {tuple(logits.shape)}")
        soft = -(teacher * torch.log_softmax(logits.float(), dim=1)).sum(1).mean()
        return self.hard_weight * hard + (1.0 - self.hard_weight) * soft


def get_model(data_config, teacher_paths=None, hard_weight=0.5, **kwargs):
    student, model_info = base.get_model(data_config, **kwargs)
    teachers = []
    for path in teacher_paths or []:
        teacher, _ = base.get_model(data_config, **kwargs)
        teacher.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        teachers.append(teacher)
    return Distilled(student, teachers), model_info


def get_loss(data_config, hard_weight=0.5, teacher_paths=None, **kwargs):
    return DistillLoss(hard_weight)
