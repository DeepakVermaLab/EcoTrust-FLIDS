import json
from pathlib import Path

from ecotrust.evaluation.publication_stage10 import _availability_masks, build_stage10_scenarios


def _registry():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs" / "publication_experiments.json").read_text())


def test_stage10_matrix_is_deduplicated_and_contains_hardest_joint():
    registry = _registry()
    assert registry["stages"]["10"]["methods"] == ["fedavg", "ecotrust_agent"]
    assert registry["stages"]["10"]["seeds"] == "primary_seeds"
    scenarios = build_stage10_scenarios(registry["stages"]["10"], registry["training"])
    assert len(scenarios) == 9
    assert scenarios[0].scenario_id == "primary"
    hardest = scenarios[-1]
    assert hardest.scenario_id == "hardest_joint"
    assert hardest.dirichlet_alpha == 0.1
    assert hardest.client_count == 50
    assert hardest.dropout_rate == 0.4
    assert hardest.straggler_rate == 0.2
    keys = {
        (s.dirichlet_alpha, s.client_count, s.dropout_rate, s.straggler_rate)
        for s in scenarios
    }
    assert len(keys) == len(scenarios)


def test_stage10_availability_is_deterministic_and_disjoint():
    clients = [f"client_{index:02d}" for index in range(20)]
    first = _availability_masks(clients, 0.4, 0.2, 123)
    second = _availability_masks(clients, 0.4, 0.2, 123)
    assert first == second
    dropped, stragglers = first
    assert dropped.isdisjoint(stragglers)
