import json
from pathlib import Path

import pytest

from ecotrust.evaluation.publication_stage12 import (
    LOCAL_BATCH_SIZE,
    _extend_device_history,
    _poison_state,
    _validate_aggregated_state,
    build_stage12_matrix,
)


def test_gpu_resume_preserves_cpu_provenance_and_batch_size():
    previous = {"device": "cpu", "batch_size": LOCAL_BATCH_SIZE}
    assert _extend_device_history(previous, "cuda") == ["cpu", "cuda"]
    assert LOCAL_BATCH_SIZE == 256


def test_repeated_gpu_resume_does_not_duplicate_device_history():
    previous = {"device_history": ["cpu", "cuda"]}
    assert _extend_device_history(previous, "cuda") == ["cpu", "cuda"]


def test_stage12_full_factorial_matrix_is_registered():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "configs" / "publication_experiments.json").read_text())
    stage = registry["stages"]["12"]
    matrix = build_stage12_matrix(stage)
    expected = (
        len(stage["attacks"])
        * len(stage["malicious_fractions"])
        * len(stage["aggregators"])
        * len(stage["matrix_seeds"])
    )
    assert len(matrix) == expected == 60
    assert {item["attack"] for item in matrix} == set(stage["attacks"])
    assert {item["aggregator"] for item in matrix} == set(stage["aggregators"])


def test_parameter_update_attack_does_not_poison_batchnorm_buffers():
    torch = pytest.importorskip("torch")
    global_state = {
        "encoder.weight": torch.tensor([1.0, 2.0]),
        "encoder_bn.running_mean": torch.tensor([0.25, -0.25]),
        "encoder_bn.running_var": torch.tensor([1.0, 1.5]),
        "encoder_bn.num_batches_tracked": torch.tensor(4),
    }
    local_state = {
        "encoder.weight": torch.tensor([2.0, 4.0]),
        "encoder_bn.running_mean": torch.tensor([0.5, -0.5]),
        "encoder_bn.running_var": torch.tensor([0.8, 1.2]),
        "encoder_bn.num_batches_tracked": torch.tensor(5),
    }

    poisoned = _poison_state(
        torch,
        local_state,
        global_state,
        "model_scaling",
        seed=42,
        parameter_keys={"encoder.weight"},
    )

    assert torch.equal(poisoned["encoder.weight"], torch.tensor([11.0, 22.0]))
    assert torch.equal(poisoned["encoder_bn.running_mean"], local_state["encoder_bn.running_mean"])
    assert torch.equal(poisoned["encoder_bn.running_var"], local_state["encoder_bn.running_var"])
    assert torch.equal(poisoned["encoder_bn.num_batches_tracked"], local_state["encoder_bn.num_batches_tracked"])
    _validate_aggregated_state(torch, poisoned)


def test_aggregated_state_rejects_negative_batchnorm_variance():
    torch = pytest.importorskip("torch")
    with pytest.raises(FloatingPointError, match="negative BatchNorm variance"):
        _validate_aggregated_state(torch, {"encoder_bn.running_var": torch.tensor([1.0, -0.1])})
