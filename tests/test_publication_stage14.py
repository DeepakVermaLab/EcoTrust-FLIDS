import json
from pathlib import Path

from ecotrust.evaluation.publication_stage14 import _MeasuredPolicy


def test_stage14_registry_requires_measured_resources_and_labeled_energy():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "configs" / "publication_experiments.json").read_text())
    stage = registry["stages"]["14"]
    required = set(stage["measurement"]["always_record"])
    assert {"wall_time_seconds", "process_cpu_seconds", "peak_rss_mb", "communication_bytes"} <= required
    assert "estimated" in stage["measurement"]["fallback"].lower()
    assert stage["comparison_seed"] == 42


def test_stage14_policy_names_are_explicit():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "configs" / "publication_experiments.json").read_text())
    for policy in registry["stages"]["14"]["policies"]:
        agent = _MeasuredPolicy(policy, 0.6, 42, 30)
        assert agent.policy == policy
