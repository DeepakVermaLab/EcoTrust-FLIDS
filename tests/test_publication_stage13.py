import json
from pathlib import Path

import pytest

from ecotrust.evaluation.publication_stage13 import LEARNED_FEATURES, _as_bool, _budget_count


def test_stage13_registry_matches_leakage_safe_features():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "configs" / "publication_experiments.json").read_text())
    stage = registry["stages"]["13"]
    assert stage["learned_features"] == list(LEARNED_FEATURES)
    assert "priority" not in LEARNED_FEATURES
    assert stage["protocol"].startswith("Leave one dataset out")


def test_budget_count_is_bounded_and_nonempty():
    assert _budget_count(0, 0.6) == 0
    assert _budget_count(1, 0.01) == 1
    assert _budget_count(20, 0.6) == 12
    assert _budget_count(20, 2.0) == 20


def test_boolean_telemetry_parsing_is_explicit():
    assert _as_bool(True) is True
    assert _as_bool("True") is True
    assert _as_bool("False") is False
    assert _as_bool(0) is False
    with pytest.raises(ValueError):
        _as_bool("ambiguous")
