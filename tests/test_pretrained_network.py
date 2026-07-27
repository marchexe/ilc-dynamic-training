import importlib.util
import unittest
from types import SimpleNamespace

import torch
import yaml

from tests.helpers import PROJECT_DIR


PRETRAINED_DIR = PROJECT_DIR / "checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut"


def _load_network_module():
    path = PROJECT_DIR / "networks/pretrained_sgv_particle_transformer.py"
    spec = importlib.util.spec_from_file_location("pretrained_sgv_particle_transformer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data_config_from_yaml(path):
    with path.open() as stream:
        raw = yaml.safe_load(stream)

    inputs = raw["inputs"]
    input_names = tuple(inputs)
    input_dicts = {
        name: [item[0] if isinstance(item, list) else item for item in config["vars"]]
        for name, config in inputs.items()
    }
    input_shapes = {
        name: (1, len(config["vars"]), config["length"])
        for name, config in inputs.items()
    }

    return SimpleNamespace(
        input_names=input_names,
        input_dicts=input_dicts,
        input_shapes=input_shapes,
        label_value=raw["labels"]["value"],
    )


class PretrainedParticleTransformerTest(unittest.TestCase):
    def test_network_loads_checkpoint_and_runs_forward(self):
        data_config = _data_config_from_yaml(PRETRAINED_DIR / "data_config.auto.yaml")
        module = _load_network_module()
        model, model_info = module.get_model(data_config)

        state = torch.load(
            PRETRAINED_DIR / "net_best_epoch_state.pt",
            map_location="cpu",
        )
        incompatible = model.load_state_dict(state, strict=True)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(model_info["input_shapes"]["pf_features"], (1, 27, 75))
        self.assertEqual(model_info["input_shapes"]["neu_features"], (1, 28, 75))

        batch_size = 2
        pf_x = torch.zeros(batch_size, 27, 75)
        neu_x = torch.zeros(batch_size, 28, 75)
        pf_v = torch.zeros(batch_size, 4, 75)
        neu_v = torch.zeros(batch_size, 4, 75)
        pf_v[:, 3, :] = 1.0
        neu_v[:, 3, :] = 1.0
        pf_mask = torch.ones(batch_size, 1, 75, dtype=torch.bool)
        neu_mask = torch.ones(batch_size, 1, 75, dtype=torch.bool)

        model.eval()
        with torch.no_grad():
            output = model(pf_x, pf_v, pf_mask, neu_x, neu_v, neu_mask)

        self.assertEqual(output.shape, (batch_size, 3))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
