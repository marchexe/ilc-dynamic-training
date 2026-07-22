import torch

from weaver.nn.model.ParticleTransformer import ParticleTransformerTagger
from weaver.utils.logger import _logger


def get_model(data_config, **kwargs):
    cfg = dict(
        pf_input_dim=len(data_config.input_dicts["pf_features"]),
        # The tagger's second collection is conventionally named `sv`; here it
        # receives the neutral-particle tensors in the same positional slots.
        sv_input_dim=len(data_config.input_dicts["neu_features"]),
        num_classes=len(data_config.label_value),
        pair_input_type="ee",
        pair_input_dim=None,
        use_pre_activation_pair=True,
        remove_self_pair=True,
        embed_dims=[128, 128, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=3,
        num_cls_layers=1,
        block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_params=[],
        activation="gelu",
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info("Model config: %s", cfg)

    model = ParticleTransformerTagger(**cfg)
    model_info = {
        "input_names": list(data_config.input_names),
        "input_shapes": {key: (1,) + shape[1:] for key, shape in data_config.input_shapes.items()},
        "output_names": ["softmax"],
        "dynamic_axes": {
            **{
                key: {0: "N", 2: "n_" + key.split("_")[0]}
                for key in data_config.input_names
            },
            "softmax": {0: "N"},
        },
    }
    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
