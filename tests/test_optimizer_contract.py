"""Full Lookahead checkpoint contract: restarted Ranger training equals uninterrupted training."""
import io
import unittest
from unittest.mock import patch

import torch

import tests.helpers  # noqa: F401  (puts scripts/ on sys.path)
from training.optimizer_contract import FORMAT, KEY, FullStateLookahead, install
from training.pbt.state.optimizer_state import set_optimizer_state_lr
from weaver.utils.nn.optimizer import ranger
from weaver.utils.nn.optimizer.lookahead import Lookahead

STEPS, CHUNKS, LR = 35, 5, 1e-3  # 7-step chunks are not multiples of Lookahead's k=6


def model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 3))


def batches():
    generator = torch.Generator().manual_seed(1234)
    return [(torch.randn(32, 8, generator=generator), torch.randint(0, 3, (32,), generator=generator)) for _ in range(STEPS)]


def train(net, optimizer, data):
    for inputs, labels in data:
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(net(inputs), labels).backward()
        optimizer.step()


def roundtrip(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


def run(lookahead, chunks):
    """Train STEPS steps in `chunks` processes, restarting from saved model and optimizer state like Weaver."""
    data, size = batches(), STEPS // chunks
    with patch.object(ranger, 'Lookahead', lookahead):
        net = model()
        optimizer = ranger.Ranger(net.parameters(), lr=LR)
        for chunk in range(chunks):
            if chunk:
                net_state, optimizer_state = roundtrip(net.state_dict()), roundtrip(optimizer.state_dict())
                net = model(seed=chunk)  # fresh process: different initial weights until the checkpoint loads
                net.load_state_dict(net_state)
                optimizer = ranger.Ranger(net.parameters(), lr=LR)
                optimizer.load_state_dict(optimizer_state)
            train(net, optimizer, data[chunk * size:(chunk + 1) * size])
    return net, optimizer


def max_difference(first, second):
    return max(float((a - b).abs().max()) for a, b in zip(first.parameters(), second.parameters()))


class FullLookaheadContractTest(unittest.TestCase):
    def test_restarted_training_equals_uninterrupted_training(self):
        reference, reference_optimizer = run(FullStateLookahead, 1)
        restarted, restarted_optimizer = run(FullStateLookahead, CHUNKS)
        for a, b in zip(reference.parameters(), restarted.parameters()):
            self.assertTrue(torch.equal(a, b))
        expected, actual = reference_optimizer.state_dict(), restarted_optimizer.state_dict()
        self.assertEqual(expected[KEY]['step_counter'], actual[KEY]['step_counter'])
        self.assertEqual(expected[KEY]['step_counter'], STEPS % 6)
        for a, b in zip(expected[KEY]['slow_weights'], actual[KEY]['slow_weights']):
            self.assertTrue(torch.equal(a, b))
        for key, slots in expected['state'].items():
            for slot, value in slots.items():
                self.assertTrue(torch.equal(torch.as_tensor(value), torch.as_tensor(actual['state'][key][slot])), (key, slot))

    def test_historical_contract_diverges_on_restart(self):
        reference, _ = run(Lookahead, 1)
        restarted, _ = run(Lookahead, CHUNKS)
        self.assertGreater(max_difference(reference, restarted), 1e-6)
        uninterrupted, _ = run(FullStateLookahead, 1)
        self.assertEqual(max_difference(reference, uninterrupted), 0.0)  # same algorithm without restarts

    def test_historical_checkpoint_loads_with_historical_behaviour(self):
        _, optimizer = run(Lookahead, 1)
        legacy = roundtrip(optimizer.state_dict())
        self.assertNotIn(KEY, legacy)
        results = []
        for lookahead in (Lookahead, FullStateLookahead):
            net = model(seed=7)
            with patch.object(ranger, 'Lookahead', lookahead):
                resumed = ranger.Ranger(net.parameters(), lr=LR)
            resumed.load_state_dict(roundtrip(legacy))  # each process reads its own copy of the file
            self.assertEqual(resumed.step_counter, 0)
            for parameter in net.parameters():
                self.assertTrue(torch.equal(resumed.state[parameter]['cached_params'], parameter.data))
            train(net, resumed, batches()[:10])
            results.append(net)
        self.assertEqual(max_difference(*results), 0.0)

    def test_engine_learning_rate_rewrite_keeps_the_lookahead_state(self):
        _, optimizer = run(FullStateLookahead, 1)
        state = roundtrip(optimizer.state_dict())
        set_optimizer_state_lr(state, 2.5e-4)
        net = model()
        with patch.object(ranger, 'Lookahead', FullStateLookahead):
            resumed = ranger.Ranger(net.parameters(), lr=LR)
        resumed.load_state_dict(state)
        self.assertEqual({group['lr'] for group in resumed.param_groups}, {2.5e-4})
        self.assertEqual(resumed.step_counter, STEPS % 6)
        for parameter, slow in zip(net.parameters(), state[KEY]['slow_weights']):
            self.assertTrue(torch.equal(resumed.state[parameter]['cached_params'], slow))

    def test_inconsistent_checkpoints_are_rejected(self):
        _, optimizer = run(FullStateLookahead, 1)
        good = roundtrip(optimizer.state_dict())
        cases = [(dict(format='lookahead-full-v0'), 'Unknown Lookahead checkpoint format'),
                 (dict(k=5), 'alpha/k'), (dict(alpha=0.4), 'alpha/k'), (dict(step_counter=6), 'step counter'),
                 (dict(slow_weights=good[KEY]['slow_weights'][:-1]), 'do not match'),
                 (dict(slow_weights=[w.flatten() for w in good[KEY]['slow_weights']]), 'do not match')]
        for change, message in cases:
            state = dict(good, **{KEY: dict(good[KEY], **change)})
            with patch.object(ranger, 'Lookahead', FullStateLookahead):
                resumed = ranger.Ranger(model().parameters(), lr=LR)
            with self.assertRaisesRegex(ValueError, message):
                resumed.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, 'pullback_momentum'):
            FullStateLookahead(torch.optim.SGD(model().parameters(), lr=LR), pullback_momentum='reset')

    def test_install_switches_weaver_ranger_and_records_the_format(self):
        with patch.object(ranger, 'Lookahead', Lookahead):
            install()
            optimizer = ranger.Ranger(model().parameters(), lr=LR)
            self.assertIsInstance(optimizer, FullStateLookahead)
            self.assertEqual(optimizer.state_dict()[KEY]['format'], FORMAT)
        self.assertIs(ranger.Lookahead, Lookahead)


if __name__ == '__main__':
    unittest.main()
