from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import traceback
from typing import Iterable

from ecotrust.config import load_config
from ecotrust.data.loaders import detect_label_columns, load_dataset_frame
from ecotrust.data.preprocess import load_preprocessed, make_binary_label
from ecotrust.io import read_json, write_json
from ecotrust.models.embedding import load_embedding_model


STAGE = 15
PREREQUISITE_STAGES = tuple(range(9, 15))
PRIMARY_METHOD = "ecotrust_agent"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_ci(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 42,
) -> dict:
    array = [float(value) for value in values if math.isfinite(float(value))]
    if not array:
        raise ValueError("A confidence interval requires at least one finite value")
    mean = float(statistics.fmean(array))
    if len(array) == 1:
        lower = upper = mean
    else:
        rng = random.Random(seed)
        samples = sorted(
            statistics.fmean(rng.choices(array, k=len(array))) for _ in range(int(resamples))
        )
        alpha = (1.0 - confidence) / 2.0
        lower_index = max(0, min(len(samples) - 1, int(alpha * (len(samples) - 1))))
        upper_index = max(0, min(len(samples) - 1, int((1.0 - alpha) * (len(samples) - 1))))
        lower, upper = float(samples[lower_index]), float(samples[upper_index])
    return {
        "n": len(array),
        "mean": mean,
        "std": float(statistics.stdev(array)) if len(array) > 1 else 0.0,
        "confidence_level": float(confidence),
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_resamples": int(resamples),
    }


def _holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, values[index] * (total - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _paired_permutation_pvalue(differences, *, resamples: int, seed: int) -> float:
    import numpy as np

    values = np.asarray(differences, dtype=float)
    if not len(values) or np.allclose(values, 0.0):
        return 1.0
    observed = abs(float(values.mean()))
    total_exact = 2 ** len(values)
    if total_exact <= resamples:
        masks = np.arange(total_exact, dtype=np.uint64)[:, None]
        bits = (masks >> np.arange(len(values), dtype=np.uint64)) & 1
        signs = np.where(bits == 0, -1.0, 1.0)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(resamples, len(values)))
    null_means = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null_means >= observed) + 1) / (len(null_means) + 1))


def _paired_statistics(proposed, baseline, *, resamples: int, seed: int) -> dict:
    import numpy as np
    from scipy import stats

    proposed = np.asarray(proposed, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if proposed.shape != baseline.shape or not len(proposed):
        raise ValueError("Paired statistics require equal nonempty arrays")
    differences = proposed - baseline
    if np.allclose(differences, 0.0):
        wilcoxon_p = ttest_p = 1.0
        rank_biserial = 0.0
    else:
        ttest_p = float(stats.ttest_rel(proposed, baseline).pvalue)
        try:
            wilcoxon_p = float(stats.wilcoxon(differences, zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        nonzero = differences[~np.isclose(differences, 0.0)]
        ranks = stats.rankdata(abs(nonzero)) if len(nonzero) else np.asarray([])
        denominator = float(ranks.sum())
        rank_biserial = (
            float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / denominator)
            if denominator
            else 0.0
        )
    permutation_p = _paired_permutation_pvalue(differences, resamples=resamples, seed=seed)
    primary_test = "wilcoxon" if len(differences) >= 5 else "paired_permutation"
    primary_p = wilcoxon_p if primary_test == "wilcoxon" else permutation_p
    standard_deviation = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    cohen_dz = float(differences.mean() / standard_deviation) if standard_deviation else 0.0
    return {
        "n_pairs": int(len(differences)),
        "paired_mean_difference": float(differences.mean()),
        "paired_median_difference": float(np.median(differences)),
        "cohen_dz": cohen_dz,
        "rank_biserial_correlation": rank_biserial,
        "paired_t_p_value": ttest_p,
        "wilcoxon_p_value": wilcoxon_p,
        "paired_permutation_p_value": permutation_p,
        "primary_test": primary_test,
        "primary_p_value": primary_p,
    }


def _stage_manifest(cfg, stage: int, run_mode: str) -> tuple[Path, dict]:
    matches = sorted(cfg.logs_root.glob(f"{stage:02d}_*_{run_mode}_manifest.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Stage {stage:02d} requires exactly one {run_mode} manifest; found {matches}")
    payload = read_json(matches[0])
    if payload.get("status") != "completed":
        raise RuntimeError(f"Stage {stage:02d} is not complete: {payload.get('status')}")
    incomplete = [key for key, value in payload.get("runs", {}).items() if value.get("status") != "completed"]
    if incomplete:
        raise RuntimeError(f"Stage {stage:02d} has incomplete runs: {incomplete[:5]}")
    return matches[0], payload


def _flatten_summary(stage: int, key: str, payload: dict, summary_path: str | None = None) -> dict:
    row = {
        "stage": stage,
        "run_key": key,
        "status": payload.get("status", "completed"),
        "dataset": payload.get("dataset"),
        "method": payload.get("method") or payload.get("policy") or payload.get("aggregator"),
        "policy": payload.get("policy"),
        "seed": payload.get("seed"),
        "summary_path": summary_path,
    }
    for name in (
        "attack", "malicious_fraction", "scenario", "alpha", "client_count",
        "dropout_rate", "straggler_rate", "energy_measurement_type", "direct_energy_source",
        "estimated_energy_j", "estimated_energy_saving_percent", "direct_energy_j",
        "communication_bytes", "wall_time_seconds", "wall_time_seconds_this_invocation",
        "process_cpu_seconds", "process_cpu_seconds_this_invocation", "peak_rss_mb",
        "controller_overhead_seconds", "controller_overhead_seconds_this_invocation", "average_accuracy",
        "backward_transfer", "forgetting", "recovery_rounds",
    ):
        if name in payload:
            row[name] = payload[name]
    final_metrics = payload.get("final_metrics", {})
    if isinstance(final_metrics, dict):
        for name, value in final_metrics.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[name] = value
    for nested_name in ("continual_metrics", "final_test_metrics"):
        nested = payload.get(nested_name, {})
        if isinstance(nested, dict):
            for name, value in nested.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[name] = value
    scenario = payload.get("scenario")
    if isinstance(scenario, dict):
        row["scenario"] = scenario.get("name")
        for name, value in scenario.items():
            if name != "name" and isinstance(value, (str, int, float, bool)):
                row[name] = value
    baseline_energy = payload.get("estimated_all_client_energy_j")
    proposed_energy = payload.get("estimated_energy_j")
    if baseline_energy is not None and proposed_energy is not None and float(baseline_energy) > 0:
        row["energy_saving_percent"] = 100.0 * (float(baseline_energy) - float(proposed_energy)) / float(baseline_energy)
    elif "estimated_energy_saving_percent" in row and "energy_saving_percent" not in row:
        row["energy_saving_percent"] = row["estimated_energy_saving_percent"]
    return row


def _experiment_ledger(manifests: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    for stage, manifest in manifests.items():
        runs = manifest.get("runs", {})
        if not runs:
            rows.append({"stage": stage, "run_key": "stage", "status": manifest.get("status")})
            continue
        for key, record in runs.items():
            summary = record.get("summary", {}) if isinstance(record, dict) else {}
            if not isinstance(summary, dict):
                summary = {}
            row = _flatten_summary(stage, key, summary)
            row["status"] = record.get("status", row["status"])
            row["skipped_on_resume"] = bool(record.get("skipped", False))
            rows.append(row)
    return rows


def _stage09_rows(cfg, run_mode: str) -> list[dict]:
    rows = []
    root = cfg.results_root / "publication" / "stage09"
    for path in sorted(root.glob("*/*/seed_*/summary.json")):
        payload = read_json(path)
        if payload.get("status") != "completed":
            continue
        rows.append(_flatten_summary(9, str(path.relative_to(root)), payload, str(path)))
    if not rows:
        raise RuntimeError("Stage 09 has no completed summary.json artifacts")
    return rows


def _stage13_rows(cfg, manifest: dict) -> list[dict]:
    import pandas as pd

    path_text = manifest.get("artifacts", {}).get("metrics")
    path = Path(path_text) if path_text else cfg.results_root / "publication" / "stage13" / "controller_generalization_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for record in pd.read_csv(path).to_dict(orient="records"):
        rows.append({
            "stage": 13,
            "run_key": f"{record['held_out_dataset']}/{record['policy']}",
            "status": "completed",
            "dataset": record["held_out_dataset"],
            "method": record["policy"],
            **record,
        })
    return rows


def _confidence_rows(stage09_rows: list[dict], registry: dict) -> list[dict]:
    import pandas as pd

    frame = pd.DataFrame(stage09_rows)
    metrics = [name for name in registry["metrics"]["primary"] + registry["metrics"]["secondary"] if name in frame]
    confidence = float(registry["metrics"]["statistics"]["confidence_level"])
    resamples = int(registry["metrics"]["statistics"]["bootstrap_resamples"])
    rows = []
    for (dataset, method), group in frame.groupby(["dataset", "method"], dropna=False):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if not len(values):
                continue
            rows.append({
                "dataset": dataset,
                "method": method,
                "metric": metric,
                **_mean_ci(values, confidence=confidence, resamples=resamples, seed=42),
            })
    return rows


def _significance_rows(stage09_rows: list[dict], registry: dict) -> list[dict]:
    import pandas as pd

    frame = pd.DataFrame(stage09_rows)
    metrics = [name for name in registry["metrics"]["primary"] if name in frame]
    baselines = sorted(set(frame["method"].dropna()) - {PRIMARY_METHOD})
    resamples = int(registry["metrics"]["statistics"]["bootstrap_resamples"])
    rows = []
    proposed = frame[frame["method"] == PRIMARY_METHOD]
    for baseline_name in baselines:
        baseline = frame[frame["method"] == baseline_name]
        paired = proposed.merge(baseline, on=["dataset", "seed"], suffixes=("_proposed", "_baseline"))
        for metric in metrics:
            left = pd.to_numeric(paired[f"{metric}_proposed"], errors="coerce")
            right = pd.to_numeric(paired[f"{metric}_baseline"], errors="coerce")
            valid = left.notna() & right.notna()
            if not valid.any():
                continue
            rows.append({
                "proposed_method": PRIMARY_METHOD,
                "baseline_method": baseline_name,
                "metric": metric,
                **_paired_statistics(left[valid], right[valid], resamples=resamples, seed=42),
            })
    adjusted = _holm_adjust(row["primary_p_value"] for row in rows)
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p_value"] = value
        row["significant_at_0_05"] = bool(value < 0.05)
    return rows


def _locked_test_attack_labels(cfg, dataset: str, run_mode: str):
    import numpy as np
    from sklearn.model_selection import train_test_split

    prep = cfg.results_root / dataset / "preprocessed" / run_mode
    manifest = read_json(prep / "preprocess_manifest.json")
    _, _, _, y_test = load_preprocessed(prep)
    frame = load_dataset_frame(dataset, cfg, nrows=manifest.get("row_cap"), max_sources=1)
    dataset_cfg = cfg.datasets[dataset]
    label_column, attack_column = detect_label_columns(frame, dataset_cfg)
    if attack_column is None:
        raise RuntimeError(f"{dataset} has no configured attack-type column")
    raw_y = make_binary_label(frame[label_column], dataset_cfg.binary_normal_values)
    indices = np.arange(len(frame))
    _, test_indices = train_test_split(
        indices, test_size=0.2, random_state=cfg.seed, stratify=raw_y
    )
    if not np.array_equal(np.asarray(raw_y)[test_indices], np.asarray(y_test)):
        raise RuntimeError(f"{dataset}: reconstructed locked test labels do not match stage 02")
    attacks = frame.iloc[test_indices][attack_column].fillna("__missing__").astype(str).to_numpy()
    return attacks, np.asarray(y_test, dtype=int)


def _model_predictions(model, X_test, model_path: Path, device: str):
    import numpy as np
    import torch

    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    predictions = []
    with torch.no_grad():
        array = np.asarray(X_test)
        for start in range(0, len(array), 2048):
            batch = torch.tensor(array[start : start + 2048], dtype=torch.float32, device=device)
            output = model(batch)
            logits = output[0] if isinstance(output, tuple) else output
            predictions.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    scores = np.concatenate(predictions)
    return (scores >= 0.5).astype(int)


def _per_attack_rows(cfg, stage09_rows: list[dict], run_mode: str, device: str) -> list[dict]:
    import numpy as np

    labels_by_dataset = {
        dataset: _locked_test_attack_labels(cfg, dataset, run_mode)
        for dataset in sorted({str(row["dataset"]) for row in stage09_rows})
    }
    rows = []
    root = cfg.results_root / "publication" / "stage09"
    for dataset in sorted(labels_by_dataset):
        prep = cfg.results_root / dataset / "preprocessed" / run_mode
        _, X_test, _, _ = load_preprocessed(prep)
        architecture_path = cfg.results_root / dataset / "embedding" / run_mode / "ecotrust_encoder.pt"
        model, _ = load_embedding_model(architecture_path, device)
        attack_labels, y_true = labels_by_dataset[dataset]
        dataset_summaries = [row for row in stage09_rows if str(row["dataset"]) == dataset]
        for summary in dataset_summaries:
            method = str(summary["method"])
            seed = int(summary["seed"])
            model_path = root / dataset / method / f"seed_{seed}" / "global_model.pt"
            if not model_path.is_file():
                raise FileNotFoundError(model_path)
            predicted = _model_predictions(model, X_test, model_path, device)
            if len(predicted) != len(y_true):
                raise RuntimeError(f"{dataset}/{method}/seed_{seed}: prediction length mismatch")
            normal_mask = y_true == 0
            tn = int(np.count_nonzero((predicted == 0) & normal_mask))
            fp = int(np.count_nonzero((predicted == 1) & normal_mask))
            rows.append({
                "dataset": dataset, "method": method, "seed": seed,
                "traffic_type": "normal", "attack_type": "__normal__",
                "support": int(normal_mask.sum()), "true_negatives": tn, "false_positives": fp,
                "false_positive_rate": fp / max(tn + fp, 1),
                "specificity": tn / max(tn + fp, 1),
                "metric_scope": "binary_detection_by_original_attack_label",
            })
            for attack_type in sorted(set(attack_labels[~normal_mask].tolist())):
                mask = (~normal_mask) & (attack_labels == attack_type)
                tp = int(np.count_nonzero((predicted == 1) & mask))
                fn = int(np.count_nonzero((predicted == 0) & mask))
                rows.append({
                    "dataset": dataset, "method": method, "seed": seed,
                    "traffic_type": "attack", "attack_type": attack_type,
                    "support": int(mask.sum()), "true_positives": tp, "false_negatives": fn,
                    "recall": tp / max(tp + fn, 1), "false_negative_rate": fn / max(tp + fn, 1),
                    "metric_scope": "binary_detection_by_original_attack_label",
                })
    return rows


def _complexity_rows(cfg, stage09_rows: list[dict], run_mode: str, device: str) -> list[dict]:
    rows = []
    root = cfg.results_root / "publication" / "stage09"
    for dataset in sorted({str(row["dataset"]) for row in stage09_rows}):
        architecture_path = cfg.results_root / dataset / "embedding" / run_mode / "ecotrust_encoder.pt"
        model, _ = load_embedding_model(architecture_path, device)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        state_bytes = sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())
        linear_macs = sum(
            module.in_features * module.out_features
            for module in model.modules()
            if module.__class__.__name__ == "Linear"
        )
        input_features = next(
            module.in_features for module in model.modules() if module.__class__.__name__ == "Linear"
        )
        methods = sorted({str(row["method"]) for row in stage09_rows if row["dataset"] == dataset})
        for method in methods:
            files = sorted((root / dataset / method).glob("seed_*/global_model.pt"))
            if not files:
                raise FileNotFoundError(f"No stage-09 model for {dataset}/{method}")
            rows.append({
                "dataset": dataset, "method": method, "input_features": int(input_features),
                "trainable_parameters": int(trainable), "state_tensor_bytes": int(state_bytes),
                "mean_model_file_bytes": float(sum(path.stat().st_size for path in files) / len(files)),
                "linear_macs_per_forward": int(linear_macs), "model_files": len(files),
            })
    return rows


def _claim_rows(manifests: dict[int, dict], significance_rows: list[dict], per_attack_rows: list[dict], ledger: list[dict]) -> list[dict]:
    any_direct_energy = any(row.get("energy_measurement_type") == "direct" for row in ledger)
    return [
        {"claim": "Matched multi-seed baseline comparison", "evidence": "stage09; confidence_intervals.csv; significance_tests.csv", "status": "supported" if significance_rows else "not_supported"},
        {"claim": "Non-IID, dropout, straggler, and scalability sensitivity", "evidence": "stage10 manifest and run summaries", "status": "supported" if manifests[10]["status"] == "completed" else "not_supported"},
        {"claim": "Continual adaptation and forgetting", "evidence": "stage11 manifest and window metrics", "status": "supported" if manifests[11]["status"] == "completed" else "not_supported"},
        {"claim": "Malicious-client robustness sensitivity", "evidence": "stage12 manifest and run summaries", "status": "supported" if manifests[12]["status"] == "completed" else "not_supported"},
        {"claim": "Leave-one-dataset-out controller generalization", "evidence": "stage13 manifest and decision metrics", "status": "supported" if manifests[13]["status"] == "completed" else "not_supported"},
        {"claim": "Resource and controller overhead", "evidence": "stage14 manifest and summaries", "status": "supported" if manifests[14]["status"] == "completed" else "not_supported"},
        {"claim": "Direct hardware energy reduction", "evidence": "stage14 energy_measurement_type", "status": "supported" if any_direct_energy else "not_supported; energy is model-estimated"},
        {"claim": "Per-attack binary detection performance", "evidence": "per_attack_metrics.csv", "status": "supported" if per_attack_rows else "not_supported"},
        {"claim": "Multiclass attack classification", "evidence": "not evaluated by this binary pipeline", "status": "not_supported"},
    ]


def _environment() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "pyarrow", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "captured_at": _utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
    }


def _checksum_files(cfg, output_dir: Path) -> list[Path]:
    project = cfg.project_root
    roots = [project / "configs", project / "docs", project / "notebooks", project / "src"]
    files: set[Path] = set()
    for root in roots:
        if root.is_dir():
            files.update(
                path for path in root.rglob("*")
                if path.is_file()
                and not path.name.startswith("._")
                and "__pycache__" not in path.parts
                and ".ipynb_checkpoints" not in path.parts
                and path.suffix != ".pyc"
            )
    files.update(
        path for path in cfg.logs_root.glob("*_manifest.json")
        if path.is_file() and not path.name.startswith("15_")
    )
    publication = cfg.results_root / "publication"
    if publication.is_dir():
        files.update(
            path for path in publication.rglob("*")
            if path.is_file() and not path.name.endswith(".tmp") and path != output_dir / "checksums.sha256"
        )
    return sorted(files)


def run_stage15(
    config_path: str | Path,
    registry_path: str | Path,
    *,
    run_mode: str = "full",
    device: str = "cpu",
) -> dict:
    import pandas as pd

    cfg = load_config(config_path)
    registry = read_json(Path(registry_path))
    stage_cfg = registry["stages"]["15"]
    output_dir = cfg.results_root / "publication" / "stage15"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.logs_root / f"15_final_statistics_and_reproducibility_{run_mode}_manifest.json"
    manifest = {
        "stage": STAGE, "status": "running", "mode": run_mode, "device": device,
        "started_at": _utc_now(), "prerequisites": {}, "artifacts": {},
    }
    write_json(manifest_path, manifest)
    try:
        manifests = {}
        for stage in PREREQUISITE_STAGES:
            path, payload = _stage_manifest(cfg, stage, run_mode)
            manifests[stage] = payload
            manifest["prerequisites"][str(stage)] = {"status": "completed", "manifest": str(path)}
        write_json(manifest_path, manifest)

        ledger = _experiment_ledger(manifests)
        ledger = [row for row in ledger if int(row["stage"]) != 13] + _stage13_rows(cfg, manifests[13])
        stage09_rows = _stage09_rows(cfg, run_mode)
        aggregate_rows = stage09_rows + [row for row in ledger if int(row["stage"]) != 9]
        confidence_rows = _confidence_rows(stage09_rows, registry)
        significance_rows = _significance_rows(stage09_rows, registry)
        per_attack_rows = _per_attack_rows(cfg, stage09_rows, run_mode, device)
        complexity_rows = _complexity_rows(cfg, stage09_rows, run_mode, device)
        claim_rows = _claim_rows(manifests, significance_rows, per_attack_rows, ledger)

        tables = {
            "experiment_ledger.csv": ledger,
            "aggregate_metrics.csv": aggregate_rows,
            "confidence_intervals.csv": confidence_rows,
            "significance_tests.csv": significance_rows,
            "claim_evidence_matrix.csv": claim_rows,
            "per_attack_metrics.csv": per_attack_rows,
            "complexity_metrics.csv": complexity_rows,
        }
        for name, rows in tables.items():
            if not rows:
                raise RuntimeError(f"Stage 15 cannot complete with empty artifact {name}")
            path = output_dir / name
            pd.DataFrame(rows).to_csv(path, index=False)
            manifest["artifacts"][name] = {"path": str(path), "rows": len(rows)}

        environment_path = output_dir / "environment.json"
        environment = _environment()
        environment.update({"config": str(config_path), "registry": str(registry_path), "run_mode": run_mode, "device": device})
        write_json(environment_path, environment)
        manifest["artifacts"]["environment.json"] = {"path": str(environment_path)}

        checksum_path = output_dir / "checksums.sha256"
        checksum_lines = []
        for path in _checksum_files(cfg, output_dir):
            checksum_lines.append(f"{_sha256(path)}  {path.relative_to(cfg.project_root)}")
        checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        manifest["artifacts"]["checksums.sha256"] = {"path": str(checksum_path), "files": len(checksum_lines)}

        missing = []
        for name in stage_cfg["required_artifacts"]:
            path = output_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(name)
        if missing:
            raise RuntimeError(f"Missing or empty registered stage-15 artifacts: {missing}")
        manifest.update({"status": "completed", "completed_at": _utc_now(), "required_artifacts": list(stage_cfg["required_artifacts"])})
        write_json(manifest_path, manifest)
        return {"manifest": str(manifest_path), **manifest}
    except BaseException as exc:
        manifest.update({
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "failed_at": _utc_now(), "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        write_json(manifest_path, manifest)
        raise
