import pytest

from ecotrust.evaluation.publication_stage15 import _flatten_summary, _holm_adjust, _mean_ci


def test_mean_ci_is_deterministic_and_contains_mean():
    first = _mean_ci([1.0, 2.0, 3.0], resamples=500, seed=7)
    second = _mean_ci([1.0, 2.0, 3.0], resamples=500, seed=7)
    assert first == second
    assert first["n"] == 3
    assert first["ci_lower"] <= first["mean"] <= first["ci_upper"]


def test_mean_ci_rejects_empty_input():
    with pytest.raises(ValueError):
        _mean_ci([])


def test_holm_adjust_is_monotonic_in_sorted_p_values():
    raw = [0.01, 0.04, 0.03]
    adjusted = _holm_adjust(raw)
    ordered = sorted(zip(raw, adjusted))
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert [value for _, value in ordered] == sorted(value for _, value in ordered)
    assert adjusted[0] == pytest.approx(0.03)


def test_flatten_summary_prefers_total_run_energy_saving():
    row = _flatten_summary(
        9,
        "dataset/method/seed_42",
        {
            "dataset": "dataset",
            "method": "method",
            "seed": 42,
            "estimated_energy_j": 60.0,
            "estimated_all_client_energy_j": 100.0,
            "final_metrics": {"estimated_energy_saving_percent": 5.0},
        },
    )
    assert row["energy_saving_percent"] == pytest.approx(40.0)
