"""Evaluation must not depend on the trimmer's unsaved training warmup."""
import importlib.util
import io
import random
import unittest
from unittest.mock import patch

import torch

from tests.helpers import PROJECT_DIR


def network_module():
    path = PROJECT_DIR / "networks/pretrained_sgv_particle_transformer.py"
    spec = importlib.util.spec_from_file_location("trimmer_checkpoint_network", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SequenceTrimmerCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.module = network_module()
        self.x = torch.arange(2 * 4 * 12, dtype=torch.float32).reshape(2, 4, 12) / 100
        self.mask = torch.arange(12)[None, None, :] < torch.tensor([3, 5])[:, None, None]

    def test_eval_is_independent_of_counter_and_preserves_rng(self):
        trimmer = self.module.SequenceTrimmer(enabled=True).eval()
        v = self.x + 1
        uu = torch.arange(2 * 2 * 12 * 12).reshape(2, 2, 12, 12)
        random_state, torch_state = random.getstate(), torch.get_rng_state()
        for counter in (0, 1, 4, 5, 100):
            trimmer._counter = counter
            output = trimmer(self.x, v, self.mask, uu)
            for actual, expected in zip(output, (self.x[..., :5], v[..., :5],
                                                self.mask[..., :5], uu[..., :5, :5])):
                self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(trimmer._counter, counter)
        self.assertEqual(random.getstate(), random_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_model_predictions_match_after_checkpoint_reload(self):
        def make_model():
            return self.module.ParticleTransformer(
                input_dim=4, num_classes=3, embed_dims=[8], pair_embed_dims=None,
                num_heads=2, num_layers=1, num_cls_layers=1, trim=True,
            )
        model = make_model()
        # Actual training-mode warmup; no weight updates are needed to expose
        # the hidden state omitted by state_dict().
        for _ in range(5):
            model.trimmer(self.x, mask=self.mask)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = make_model()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        self.assertEqual(restored.trimmer._counter, 0)
        model.eval()
        restored.eval()
        with torch.no_grad():
            expected = model(self.x, mask=self.mask)
            actual = restored(self.x, mask=self.mask)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(restored.trimmer(self.x, mask=self.mask)[0],
                                    model.trimmer(self.x, mask=self.mask)[0]))

    def test_training_warmup_and_random_trimming_are_unchanged(self):
        trimmer = self.module.SequenceTrimmer(enabled=True).train()
        for count in range(5):
            self.assertEqual(trimmer(self.x, mask=self.mask)[0].shape[-1], 12)
            self.assertEqual(trimmer._counter, count + 1)
        with patch.object(self.module.random, "uniform", return_value=1) as uniform:
            trimmed, _, mask, _ = trimmer(self.x, mask=self.mask)
        uniform.assert_called_once_with(*trimmer.target)
        self.assertEqual(trimmed.shape[-1], 5)
        self.assertTrue(torch.equal(mask.sum(-1), self.mask.sum(-1)))
        for index in range(2):
            self.assertTrue(torch.equal(trimmed[index, :, mask[index, 0]].sort(-1).values,
                                        self.x[index, :, self.mask[index, 0]].sort(-1).values))

    def test_empty_and_disabled_trimming(self):
        trimmer = self.module.SequenceTrimmer(enabled=True).eval()
        self.assertEqual(trimmer(self.x, mask=torch.zeros_like(self.mask))[0].shape[-1], 1)
        trimmer.enabled = False
        self.assertTrue(torch.equal(trimmer(self.x, mask=self.mask)[0], self.x))


if __name__ == "__main__":
    unittest.main()
