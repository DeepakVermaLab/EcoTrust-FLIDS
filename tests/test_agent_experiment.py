import pytest

from ecotrust.agent.experiment import _comparison, _validate_baseline


def test_validate_baseline_requires_completed_all_client_protocol():
    baseline = {
        "status": "completed",
        "selection_policy": "all",
        "completed_rounds": 5,
    }
    _validate_baseline(baseline, "dataset", rounds=5)

    with pytest.raises(RuntimeError, match="all-client"):
        _validate_baseline({**baseline, "selection_policy": "agent"}, "dataset", rounds=5)

    with pytest.raises(RuntimeError, match="expected 5"):
        _validate_baseline({**baseline, "completed_rounds": 4}, "dataset", rounds=5)


def test_comparison_uses_matched_energy_and_metric_deltas():
    baseline = {
        "final_metrics": {"f1": 0.80, "accuracy": 0.85},
        "proposed_energy_j": 100.0,
        "elapsed_seconds": 12.0,
    }
    agentic = {
        "final_metrics": {"f1": 0.82, "accuracy": 0.86},
        "proposed_energy_j": 65.0,
        "elapsed_seconds": 9.0,
        "completed_rounds": 5,
    }

    result = _comparison(baseline, agentic, budget_fraction=0.6)

    assert result["matched_baseline_policy"] == "all"
    assert result["rounds"] == 5
    assert result["f1_delta"] == pytest.approx(0.02)
    assert result["accuracy_delta"] == pytest.approx(0.01)
    assert result["energy_saving_percent"] == pytest.approx(35.0)
