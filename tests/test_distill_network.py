"""networks/distill_particle_transformer.py: the loss and the student-only checkpoints."""
import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'networks' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tiny(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.copy_(torch.tensor(bias))

    def no_weight_decay(self):
        return set()

    def forward(self, x):
        return self.linear(x)


class DistillTest(unittest.TestCase):
    def setUp(self):
        self.d = load('distill_particle_transformer')

    def test_loss_mixes_hard_and_soft_terms(self):
        teachers = [Tiny([2.0, 0.0, 0.0]), Tiny([0.0, 2.0, 0.0])]
        model = self.d.Distilled(Tiny([0.0, 0.0, 0.0]), teachers)
        model.train()
        x, label = torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1])
        logits = model(x)
        loss = self.d.DistillLoss(0.5)(logits, label)
        probs = torch.stack([torch.softmax(t(x), 1) for t in teachers]).mean(0)
        hard = torch.nn.functional.cross_entropy(logits, label)
        soft = -(probs * torch.log_softmax(logits, 1)).sum(1).mean()
        self.assertTrue(torch.allclose(loss, 0.5 * hard + 0.5 * soft))
        loss.backward()
        self.assertTrue(all(p.grad is None for t in teachers for p in t.parameters()))

    def test_validation_uses_the_hard_loss_only(self):
        model = self.d.Distilled(Tiny([0.0, 0.0, 0.0]), [Tiny([2.0, 0.0, 0.0])])
        model.eval()
        x, label = torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1])
        logits = model(x)
        self.assertTrue(torch.allclose(self.d.DistillLoss(0.5)(logits, label),
                                       torch.nn.functional.cross_entropy(logits, label)))

    def test_checkpoint_and_parameters_hold_the_student_only(self):
        student = Tiny([0.0, 0.0, 0.0])
        model = self.d.Distilled(student, [Tiny([2.0, 0.0, 0.0])])
        self.assertEqual(set(model.state_dict()), set(student.state_dict()))
        self.assertEqual(len(list(model.parameters())), len(list(student.parameters())))
        other = Tiny([1.0, 1.0, 1.0])
        model.load_state_dict(other.state_dict())
        self.assertTrue(torch.equal(student.linear.bias, other.linear.bias))


if __name__ == '__main__':
    unittest.main()
