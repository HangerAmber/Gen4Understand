from pathlib import Path

import pytest

from gen4understand.cli import main
from gen4understand.demo import run_demo
from gen4understand.generation import plan_generation
from gen4understand.io import read_jsonl
from gen4understand.pipeline import build_dataset
from gen4understand.schema import Event, Unit
from gen4understand.verifier import Calibrator, Candidate


def test_end_to_end_cpu_fixture(tmp_path):
    report = run_demo(tmp_path)
    assert report["accepted_events"] == 2
    records = list(read_jsonl(tmp_path / "dataset/train.jsonl"))
    assert {r["level"] for r in records} == {"atom", "act", "sub", "task"}
    assert all(Path(r["video"]).is_file() for r in records)
    assert all(r["answer"] for r in records)
    main(["validate", str(tmp_path / "events.jsonl")])
    events = [Event.from_dict(x) for x in read_jsonl(tmp_path / "events.jsonl")]
    requests = plan_generation(events, tmp_path / "anchors")
    assert len(requests) == 8 and len({r["seed"] for r in requests}) == 8
    assert all(Path(r["end_frame"]).is_file() for r in requests)
    candidates = [Candidate.from_dict(x) for x in read_jsonl(tmp_path / "candidates.jsonl")]
    selected = Calibrator((0.0,) * 5, (1.0,) * 5).select(candidates)
    selected.pop("event-1")
    units = [Unit("task", "task", ("event-0", "event-1"))]
    rows, skipped = build_dataset(events, units, selected, tmp_path / "partial")
    assert len(skipped) == 1
    assert {r["level"] for r in rows} == {"atom"}


def test_training_loss_masks_prompt_and_applies_weight_once():
    torch = pytest.importorskip("torch")
    from gen4understand.training import weighted_response_loss

    logits = torch.zeros(2, 4, 5, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 2], [-100, -100, 1, 2]])
    loss = weighted_response_loss(logits, labels, ["atom", "task"])
    assert loss.item() == pytest.approx(0.6 * torch.log(torch.tensor(5.0)).item())
    loss.backward()
    assert logits.grad[:, 0].abs().sum() == 0
    assert logits.grad[:, 1:3].abs().sum() > 0


def test_tiny_qwen_lora_freeze_gradient_and_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from safetensors.torch import load_file

    from gen4understand.training import attach_lora

    config = transformers.Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "fullatt_block_indexes": [0],
        },
    )
    model = attach_lora(transformers.Qwen2_5_VLForConditionalGeneration(config), rank=2, alpha=4)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert any("merger" in name for name in trainable)
    assert not any("visual" in name and "merger" not in name for name in trainable)
    ids = torch.tensor([[1, 2, 3, 4]])
    model(input_ids=ids, labels=ids).loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )
    model.save_pretrained(tmp_path)
    checkpoint = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert any("merger" in name for name in checkpoint)
    assert any("lora_" in name for name in checkpoint)
