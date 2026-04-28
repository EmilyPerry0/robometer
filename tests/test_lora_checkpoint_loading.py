import sys
from types import ModuleType, SimpleNamespace

import torch
from safetensors.torch import save_file

fake_unsloth = ModuleType("unsloth")
fake_unsloth.FastVisionModel = object()
sys.modules.setdefault("unsloth", fake_unsloth)

fake_collators = ModuleType("robometer.data.collators")
fake_collators.BaseCollator = object
fake_collators.ReWiNDBatchCollator = object
fake_collators.RBMBatchCollator = object
sys.modules.setdefault("robometer.data.collators", fake_collators)

fake_datasets = ModuleType("robometer.data.datasets")
fake_datasets.RBMDataset = object
fake_datasets.StrategyFirstDataset = object
fake_datasets.BaseDataset = object
fake_datasets.RepeatedDataset = object
sys.modules.setdefault("robometer.data.datasets", fake_datasets)

fake_custom_eval = ModuleType("robometer.data.datasets.custom_eval")
fake_custom_eval.CustomEvalDataset = object
sys.modules.setdefault("robometer.data.datasets.custom_eval", fake_custom_eval)

fake_models = ModuleType("robometer.models")
fake_models.RBM = object
fake_models.ReWiNDTransformer = object
fake_models.ReWINDTransformerConfig = object
sys.modules.setdefault("robometer.models", fake_models)

from robometer.utils.setup_utils import _load_checkpoint_weights_from_safetensors  # noqa: E402


class FakeProgressHead:
    def __init__(self):
        self.weight = torch.tensor([0.0])


class FakeModel:
    def __init__(self):
        self.progress_head = [FakeProgressHead()]
        self.adapter_key = "model.language_model.base_model.model.layer.lora_A.default.weight"
        self.adapter_weight = torch.tensor([1.0])

    def state_dict(self):
        return {
            "progress_head.0.weight": self.progress_head[0].weight,
            self.adapter_key: self.adapter_weight,
        }

    def load_state_dict(self, state_dict, strict=False):
        if "progress_head.0.weight" in state_dict:
            self.progress_head[0].weight = state_dict["progress_head.0.weight"]
        if self.adapter_key in state_dict:
            self.adapter_weight = state_dict[self.adapter_key]
        return [], []


def test_load_checkpoint_skips_adapter_change_verification_when_adapters_already_loaded(tmp_path):
    save_file(
        {
            "progress_head.0.weight": torch.tensor([2.0]),
            "base_model.model.layer.lora_A.default.weight": torch.tensor([3.0]),
        },
        tmp_path / "custom_heads.safetensors",
    )

    model = FakeModel()

    _load_checkpoint_weights_from_safetensors(
        model,
        str(tmp_path),
        SimpleNamespace(use_peft=True),
        load_adapters=False,
    )

    assert torch.equal(model.progress_head[0].weight, torch.tensor([2.0]))
    assert torch.equal(model.adapter_weight, torch.tensor([1.0]))
