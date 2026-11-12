#!/usr/bin/env python3
"""Compile accepted EcoTrust evidence into the static demonstrator bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
DIST = ROOT / "dist"
DOWNLOADS = DIST / "downloads"

DATASETS = ("Edge-IIoTset", "TON_IoT", "CICIoT2023")
POLICIES = (
    "all_clients",
    "random_budget",
    "round_robin",
    "static_score",
    "telemetry_heuristic",
    "learned_policy",
)

MANIFESTS = {
    "09": "09_recent_baselines_and_multiseed_full_manifest.json",
    "10": "10_non_iid_dropout_scalability_full_manifest.json",
    "11": "11_temporal_drift_and_forgetting_full_manifest.json",
    "12": "12_malicious_client_robustness_full_manifest.json",
    "13": "13_cross_dataset_controller_generalization_full_manifest.json",
    "14": "14_resource_energy_policy_comparison_full_manifest.json",
    "15": "15_final_statistics_and_reproducibility_full_manifest.json",
}

DOWNLOAD_FILES = (
    "aggregate_metrics.csv",
    "claim_evidence_matrix.csv",
    "complexity_metrics.csv",
    "confidence_intervals.csv",
    "environment.json",
    "per_attack_metrics.csv",
    "significance_tests.csv",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def scalar(value: str | int | float | None) -> object:
    if value in (None, ""):
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return value
    return int(value_float) if value_float.is_integer() else value_float


def compact_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [{key: scalar(value) for key, value in row.items()} for row in rows]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifests() -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for stage, filename in MANIFESTS.items():
        path = EVIDENCE / "manifests" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing accepted manifest: {path}")
        manifest = read_json(path)
        if manifest.get("status") != "completed" or manifest.get("error"):
            raise RuntimeError(f"Stage {stage} is not accepted as completed: {path}")
        output[stage] = {
            "status": manifest.get("status"),
            "mode": manifest.get("mode", "full"),
            "device": manifest.get("device"),
            "completed_at": manifest.get("completed_at"),
            "filename": filename,
        }
    return output


def policy_data() -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for policy in POLICIES:
            run = EVIDENCE / "stage14" / dataset / policy / "seed_42"
            summary = read_json(run / "summary.json")
            overhead = read_json(run / "controller_overhead.json")
            rounds = compact_rows(read_csv(run / "federated_round_metrics.csv"))
            if summary.get("status") != "completed" or len(rounds) != 30:
                raise RuntimeError(f"Incomplete Stage 14 policy evidence: {run}")
            metrics = summary["final_metrics"]
            output[dataset][policy] = {
                "status": summary["status"],
                "seed": summary["seed"],
                "device": summary["device"],
                "energy_measurement_type": summary["energy_measurement_type"],
                "energy_measurement_scope": summary["energy_measurement_scope"],
                "estimated_energy_j": summary["estimated_energy_j"],
                "estimated_energy_saving_percent": summary["estimated_energy_saving_percent"],
                "communication_bytes": summary["communication_bytes"],
                "wall_time_seconds": summary["wall_time_seconds"],
                "peak_rss_mb": summary["peak_rss_mb"],
                "controller_overhead_seconds": overhead["controller_overhead_seconds"],
                "final_metrics": metrics,
                "rounds": rounds,
            }
    return output


def copy_downloads(manifests: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for filename in DOWNLOAD_FILES:
        source = EVIDENCE / "stage15" / filename
        target = DOWNLOADS / filename
        shutil.copy2(source, target)
        copied.append(target)
    manifest_dir = DOWNLOADS / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    for stage in manifests:
        filename = str(manifests[stage]["filename"])
        target = manifest_dir / filename
        shutil.copy2(EVIDENCE / "manifests" / filename, target)
        copied.append(target)
    records = [
        {
            "file": str(path.relative_to(DIST)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(copied)
    ]
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "accepted EcoTrust-FLIDS Stage 09-15 artifacts",
        "files": records,
    }
    provenance_path = DOWNLOADS / "evidence_manifest.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    records.append(
        {
            "file": str(provenance_path.relative_to(DIST)),
            "bytes": provenance_path.stat().st_size,
            "sha256": sha256(provenance_path),
        }
    )
    return records


def main() -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    manifests = validate_manifests()
    policies = policy_data()
    stage15 = EVIDENCE / "stage15"
    claims = read_csv(stage15 / "claim_evidence_matrix.csv")
    significance = compact_rows(read_csv(stage15 / "significance_tests.csv"))
    intervals = compact_rows(read_csv(stage15 / "confidence_intervals.csv"))
    attacks = compact_rows(read_csv(stage15 / "per_attack_metrics.csv"))
    complexity = compact_rows(read_csv(stage15 / "complexity_metrics.csv"))
    downloads = copy_downloads(manifests)

    payload = {
        "meta": {
            "title": "EcoTrust-FLIDS Research Demonstrator",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "evidence_boundary": "Accepted Stage 09-15 artifacts; custom scenarios are modeled what-if simulations.",
            "datasets": list(DATASETS),
            "stages_verified": list(manifests),
            "experiment_ledger_rows": 447,
            "confidence_interval_rows": len(intervals),
            "significance_test_rows": len(significance),
            "checksum_entries": 2394,
        },
        "manifests": manifests,
        "policies": policies,
        "claims": claims,
        "significance": significance,
        "confidence_intervals": intervals,
        "per_attack": attacks,
        "complexity": complexity,
        "downloads": downloads,
        "controller": {
            "min_trust": 0.55,
            "min_energy_j": 120,
            "trust_weight": 0.45,
            "energy_weight": 0.20,
            "quality_weight": 0.25,
            "drift_boost": 0.35,
            "latency_weight": 0.10,
            "energy_normalizer_j": 1200,
            "latency_normalizer_ms": 1000,
        },
    }
    output = "window.ECOTRUST_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    (DIST / "data.js").write_text(output, encoding="utf-8")
    print("SITE_DATA_STATUS: built")
    print(f"datasets: {len(policies)}")
    print(f"policy endpoints: {sum(len(value) for value in policies.values())}")
    print(f"per-attack rows: {len(attacks)}")
    print(f"downloads: {len(downloads)}")


if __name__ == "__main__":
    main()

