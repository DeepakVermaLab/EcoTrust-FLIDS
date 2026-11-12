from pathlib import Path

from ecotrust.config import load_config
from ecotrust.data.audit import audit_datasets


def test_config_loads():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "project.json")
    assert cfg.project_root.name == "IMPLEMENTATION"
    assert set(cfg.datasets) == {"Edge-IIoTset", "TON_IoT", "CICIoT2023"}


def test_audit_structure():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "project.json")
    report = audit_datasets(cfg)
    assert "datasets" in report
    assert set(report["datasets"]) == {"Edge-IIoTset", "TON_IoT", "CICIoT2023"}
