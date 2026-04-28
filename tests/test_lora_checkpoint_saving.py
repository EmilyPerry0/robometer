import json
from types import SimpleNamespace

import torch

from robometer.utils import save


class FakePeftModel:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, path):
        self.saved_to = path
        (path / "adapter_config.json").write_text("{}")
        (path / "adapter_model.safetensors").write_bytes(b"adapter")


class FakeInnerModel:
    def __init__(self, language_model):
        self.language_model = language_model
        self.visual = object()


class FakeRBM:
    def __init__(self, peft_model):
        self.model = FakeInnerModel(peft_model)

    def named_parameters(self):
        yield "progress_head.weight", torch.nn.Parameter(torch.ones(1, 2))

    def named_buffers(self):
        return iter(())


class FakeTrainer:
    def __init__(self, model, output_dir):
        self.model = model
        self.args = SimpleNamespace(output_dir=str(output_dir), should_save=False)
        self.save_model_called = False

    def save_model(self, _path):
        self.save_model_called = True


def test_save_trainer_checkpoint_files_saves_nested_language_model_lora(tmp_path, monkeypatch):
    monkeypatch.setattr(save, "PeftModel", FakePeftModel)
    peft_model = FakePeftModel()
    trainer = FakeTrainer(FakeRBM(peft_model), tmp_path)
    ckpt_dir = tmp_path / "checkpoint"

    save._save_trainer_checkpoint_files(trainer, trainer.args, ckpt_dir)

    assert peft_model.saved_to == ckpt_dir
    assert not trainer.save_model_called
    assert (ckpt_dir / "adapter_config.json").exists()
    assert (ckpt_dir / "adapter_model.safetensors").exists()
    assert (ckpt_dir / "custom_heads.safetensors").exists()
    assert json.loads((ckpt_dir / "peft_target_module.json").read_text()) == {
        "target_module": "language_model",
        "adapter_dir": ".",
    }
