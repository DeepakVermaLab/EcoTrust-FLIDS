from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import traceback
from typing import Iterable

from ecotrust.config import load_config, run_mode_settings
from ecotrust.io import read_json, write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int | None:
    if path.suffix == ".csv":
        with path.open("rb") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            return int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            return None
    return None


def _json_status(path: Path) -> tuple[str | None, int | None]:
    if path.suffix != ".json":
        return None, None
    try:
        payload = read_json(path)
    except Exception:
        return "invalid_json", None
    if not isinstance(payload, dict):
        return None, None
    rounds = payload.get("completed_rounds")
    return payload.get("status"), int(rounds) if rounds is not None else None


def _project_files(project_root: Path) -> list[tuple[str, str, Path]]:
    files: list[tuple[str, str, Path]] = []
    for relative in (
        "pyproject.toml",
        "requirements.txt",
        "requirements-core.txt",
        "requirements-ml.txt",
        "configs/project.json",
        "configs/project_colab.json",
    ):
        files.append(("project", relative, project_root / relative))
    for directory, stage in (("src", "source"), ("notebooks", "notebook"), ("tests", "test")):
        root = project_root / directory
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.startswith("._") or "__pycache__" in path.parts:
                continue
            if path.suffix not in {".py", ".ipynb"}:
                continue
            files.append((stage, str(path.relative_to(project_root)), path))
    return files


def _result_files(cfg, dataset: str, run_mode: str) -> list[tuple[str, str, Path]]:
    root = cfg.results_root / dataset
    files: list[tuple[str, str, Path]] = []
    for name in (
        "X_train.parquet",
        "X_test.parquet",
        "y_train.parquet",
        "y_test.parquet",
        "preprocessor.joblib",
        "client_splits.json",
        "preprocess_manifest.json",
    ):
        files.append(("02_preprocessing", name, root / "preprocessed" / run_mode / name))
    for name in (
        "logistic_regression.joblib",
        "random_forest.joblib",
        "metrics.csv",
        "metrics.json",
        "REPORT.md",
    ):
        files.append(("03_baselines", name, root / "baselines" / run_mode / name))
    for name in (
        "ecotrust_encoder.pt",
        "Z_train.parquet",
        "Z_test.parquet",
        "embedding_history.json",
    ):
        files.append(("04_embedding", name, root / "embedding" / run_mode / name))
    for stage, folder in (("05_federated", "federated"), ("06_agentic", "agentic")):
        for name in (
            "federated_summary.json",
            "federated_global_model.pt",
            "federated_round_metrics.csv",
            "federated_round_history.csv",
            "federated_progress.json",
        ):
            files.append((stage, name, root / folder / run_mode / name))
    files.append(
        (
            "06_agentic",
            "baseline_agentic_comparison.json",
            root / "agentic" / run_mode / "baseline_agentic_comparison.json",
        )
    )
    for variant in ("no_trust", "no_energy", "no_drift"):
        for name in (
            "federated_summary.json",
            "federated_global_model.pt",
            "federated_round_metrics.csv",
            "federated_round_history.csv",
            "federated_progress.json",
        ):
            files.append(
                (
                    f"07_ablation_{variant}",
                    name,
                    root / "ablation" / variant / run_mode / name,
                )
            )
    return files


def run_reproducibility_audit(
    config_path: str | Path,
    *,
    run_mode: str = "full",
    datasets: Iterable[str] = ("Edge-IIoTset", "TON_IoT", "CICIoT2023"),
) -> dict:
    """Verify required code and result artifacts without copying raw datasets."""

    import pandas as pd

    cfg = load_config(config_path)
    settings = run_mode_settings(cfg, run_mode)
    rounds = int(settings["rounds"])
    output_dir = cfg.results_root / "reproducibility" / run_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.logs_root / f"08_reproducibility_audit_{run_mode}_manifest.json"
    manifest = {
        "stage": 8,
        "status": "running",
        "mode": run_mode,
        "started_at": _utc_now(),
        "raw_datasets_copied": False,
        "datasets": list(datasets),
    }
    write_json(manifest_path, manifest)

    try:
        candidates = _project_files(cfg.project_root)
        candidates.append(("01_dataset_audit", "dataset_audit.json", cfg.results_root / "dataset_audit.json"))
        for dataset in datasets:
            candidates.extend(_result_files(cfg, dataset, run_mode))
        for name in (
            "ablation_summary.csv",
            "controlled_prior_drift_metrics.csv",
        ):
            candidates.append(
                ("07_ablation_drift", name, cfg.results_root / "ablation_drift" / run_mode / name)
            )
        for name in (
            f"05_incremental_federated_learning_{run_mode}_manifest.json",
            f"06_agentic_energy_controller_{run_mode}_manifest.json",
            f"07_ablation_and_drift_analysis_{run_mode}_manifest.json",
        ):
            candidates.append(("stage_manifest", name, cfg.logs_root / name))

        rows = []
        missing = []
        invalid = []
        for stage, artifact, path in candidates:
            exists = path.is_file()
            row = {
                "stage": stage,
                "artifact": artifact,
                "path": str(path),
                "exists": exists,
                "size_bytes": int(path.stat().st_size) if exists else None,
                "sha256": _sha256(path) if exists else None,
                "row_count": _row_count(path) if exists else None,
            }
            status, completed_rounds = _json_status(path) if exists else (None, None)
            row["json_status"] = status
            row["completed_rounds"] = completed_rounds
            rows.append(row)
            if not exists:
                missing.append(str(path))
            if artifact == "federated_summary.json" and (
                status != "completed" or completed_rounds != rounds
            ):
                invalid.append(
                    {"path": str(path), "status": status, "completed_rounds": completed_rounds}
                )
            if stage == "stage_manifest" and status != "completed":
                invalid.append({"path": str(path), "status": status})

        inventory_path = output_dir / "artifact_inventory.csv"
        pd.DataFrame(rows).to_csv(inventory_path, index=False)
        checksums_path = output_dir / "checksums.sha256"
        checksums_path.write_text(
            "".join(
                f"{row['sha256']}  {row['path']}\n"
                for row in rows
                if row["exists"] and row["sha256"]
            ),
            encoding="utf-8",
        )
        package_names = (
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "torch",
            "pyarrow",
            "joblib",
        )
        environment = {
            "captured_at": _utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {},
        }
        for package in package_names:
            try:
                environment["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                environment["packages"][package] = None
        environment_path = output_dir / "environment.json"
        write_json(environment_path, environment)
        manifest.update(
            {
                "status": "completed" if not missing and not invalid else "failed",
                "completed_at": _utc_now(),
                "artifact_count": len(rows),
                "missing_count": len(missing),
                "invalid_count": len(invalid),
                "missing": missing,
                "invalid": invalid,
                "artifacts": {
                    "inventory": str(inventory_path),
                    "checksums": str(checksums_path),
                    "environment": str(environment_path),
                },
            }
        )
        write_json(manifest_path, manifest)
        if missing or invalid:
            raise RuntimeError(
                f"Reproducibility audit failed: {len(missing)} missing and {len(invalid)} invalid artifacts"
            )
        return {"manifest": str(manifest_path), **manifest}
    except BaseException as exc:
        if manifest.get("status") != "failed":
            manifest.update(
                {
                    "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    "failed_at": _utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            write_json(manifest_path, manifest)
        raise
