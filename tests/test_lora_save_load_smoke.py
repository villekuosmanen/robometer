import sys
from types import ModuleType, SimpleNamespace

import torch
from peft import PeftModel
from transformers import GPT2Config, GPT2LMHeadModel

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

from robometer.utils.save import save_final_checkpoint  # noqa: E402
from robometer.utils.setup_utils import (  # noqa: E402
    _inspect_checkpoint_for_peft,
    _load_custom_heads_from_safetensors,
    model_has_peft,
    setup_peft_model,
)


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = GPT2Config(
            vocab_size=32,
            n_positions=8,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        self.language_model = GPT2LMHeadModel(config)
        self.visual = torch.nn.Linear(2, 2)


class TinyRBM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.progress_head = torch.nn.Sequential(torch.nn.Linear(4, 1))


class TinyTrainer:
    def __init__(self, model, output_dir):
        self.model = model
        self.args = SimpleNamespace(output_dir=str(output_dir), should_save=False)

    def save_model(self, _path):
        raise AssertionError("PEFT smoke should use adapter save path, not full model save")


def _peft_cfg():
    return SimpleNamespace(
        r=2,
        lora_alpha=4,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        bias="none",
        peft_vision_encoder=False,
    )


def _adapter_tensors(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }


def test_tiny_lora_training_save_and_targeted_load(tmp_path):
    torch.manual_seed(0)
    peft_cfg = _peft_cfg()
    model = setup_peft_model(TinyRBM(), peft_cfg)
    assert model_has_peft(model)

    before_train = _adapter_tensors(model.model.language_model)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    input_ids = torch.randint(0, 32, (2, 6))
    loss = model.model.language_model(input_ids=input_ids, labels=input_ids).loss
    loss = loss + model.progress_head(torch.ones(1, 4)).sum()
    loss.backward()
    optimizer.step()

    after_train = _adapter_tensors(model.model.language_model)
    assert any(not torch.equal(before_train[name], after_train[name]) for name in before_train)

    ckpt_dir = tmp_path / "lora-ckpt"
    save_final_checkpoint(TinyTrainer(model, tmp_path), str(ckpt_dir), step=1)

    peft_info = _inspect_checkpoint_for_peft(str(ckpt_dir))
    assert peft_info["has_adapter_files"]
    assert peft_info["target_module"] == "language_model"
    assert (ckpt_dir / "adapter_model.safetensors").exists()
    assert (ckpt_dir / "adapter_config.json").exists()
    assert (ckpt_dir / "peft_target_module.json").exists()
    assert (ckpt_dir / "custom_heads.safetensors").exists()

    loaded = TinyRBM()
    loaded.model.language_model = PeftModel.from_pretrained(
        loaded.model.language_model,
        peft_info["adapter_load_path"],
    )
    assert _load_custom_heads_from_safetensors(loaded, str(ckpt_dir))

    loaded_adapters = _adapter_tensors(loaded.model.language_model)
    for name, trained_value in after_train.items():
        assert torch.allclose(loaded_adapters[name], trained_value)
