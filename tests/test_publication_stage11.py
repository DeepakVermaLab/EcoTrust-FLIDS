import json
from pathlib import Path

import pytest

from ecotrust.evaluation.publication_stage11 import METHODS, _continual_statistics
from ecotrust.federated.client import _singleton_safe_batch_size


def test_stage11_registry_locks_methods_and_training_only_validation():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "configs" / "publication_experiments.json").read_text())
    stage = registry["stages"]["11"]
    assert stage["methods"] == list(METHODS)
    assert stage["seeds"] == "primary_seeds"
    assert stage["validation_fraction_per_window"] == pytest.approx(0.2)
    assert registry["data"]["test_policy"].startswith("Evaluate the locked final configuration once")


def test_continual_statistics_matches_known_matrix():
    rows = []
    matrix = [[0.8], [0.7, 0.9], [0.6, 0.85, 0.95]]
    for after, values in enumerate(matrix):
        for task, value in enumerate(values):
            rows.append({"after_window": after, "eval_window": task, "macro_f1": value})
    result = _continual_statistics(rows, windows=3, rounds_per_window=6)
    assert result["average_accuracy"] == pytest.approx((0.6 + 0.85 + 0.95) / 3)
    assert result["backward_transfer"] == pytest.approx((-0.2 - 0.05) / 2)
    assert result["forgetting"] == pytest.approx((0.2 + 0.05) / 2)


def test_local_batch_size_retains_samples_without_singleton_tail():
    assert _singleton_safe_batch_size(256, 256) == 256
    assert _singleton_safe_batch_size(257, 256) == 255
    assert 513 % _singleton_safe_batch_size(513, 256) != 1
    assert _singleton_safe_batch_size(20, 256) == 20

    with pytest.raises(ValueError, match="at least two samples"):
        _singleton_safe_batch_size(1, 256)
    with pytest.raises(ValueError, match="batch_size"):
        _singleton_safe_batch_size(20, 1)
