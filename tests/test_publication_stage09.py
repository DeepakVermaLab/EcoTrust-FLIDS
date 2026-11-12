import json
from pathlib import Path

import pytest

from ecotrust.evaluation.publication_stage09 import METHODS, _method_settings, _server_update


def test_publication_registry_is_frozen_and_does_not_tune_on_test():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "configs" / "publication_experiments.json").read_text())

    assert registry["status"] == "frozen_before_stage_09"
    assert registry["data"]["validation_policy"].startswith("Create validation subsets from training")
    assert registry["integrity"]["test_set_hyperparameter_selection"] is False
    assert registry["stages"]["09"]["methods"] == list(METHODS)
    assert registry["stages"]["09"]["fedadam"]["server_learning_rate"] == 0.01
    assert registry["stages"]["09"]["protocol_amendment"]["test_data_used_for_selection"] is False
    assert len(registry["randomness"]["primary_seeds"]) >= 3


def test_stage09_method_protocols_are_matched_and_distinct():
    fedavg = _method_settings("fedavg")
    fedprox = _method_settings("fedprox")
    fedadam = _method_settings("fedadam")
    trust = _method_settings("trust_weighted_all_clients")
    proposed = _method_settings("ecotrust_agent")

    assert fedavg["selection"] == fedprox["selection"] == fedadam["selection"] == "all"
    assert fedprox["proximal_mu"] > 0
    assert fedadam["server_optimizer"] == "fedadam"
    assert trust["trust_weighted"] is True and trust["selection"] == "all"
    assert proposed["trust_weighted"] is True and proposed["selection"] == "agent"

    with pytest.raises(ValueError, match="Unknown stage-09 method"):
        _method_settings("invented_method")


def test_fedadam_first_step_is_finite_and_bounded():
    torch = pytest.importorskip("torch")
    current = {"weight": torch.zeros(4), "counter": torch.tensor(3, dtype=torch.int64)}
    averaged = {
        "weight": torch.tensor([1000.0, -1000.0, 0.5, -0.5]),
        "counter": torch.tensor(4, dtype=torch.int64),
    }
    config = {
        "server_learning_rate": 0.01,
        "beta1": 0.9,
        "beta2": 0.99,
        "tau": 0.001,
    }

    updated, state = _server_update(
        torch, current, averaged, {}, "fedadam", fedadam_config=config
    )

    assert torch.isfinite(updated["weight"]).all()
    assert float(updated["weight"].abs().max()) <= 0.011
    assert updated["counter"].item() == 3
    assert state["config"] == config


def test_fedadam_rejects_invalid_server_learning_rate():
    torch = pytest.importorskip("torch")
    state = {"weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="Invalid FedAdam"):
        _server_update(
            torch,
            state,
            state,
            {},
            "fedadam",
            fedadam_config={"server_learning_rate": 0.0},
        )
