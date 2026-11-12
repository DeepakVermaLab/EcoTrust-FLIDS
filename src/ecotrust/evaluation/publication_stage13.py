from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Iterable

from ecotrust.config import load_config
from ecotrust.energy.simulator import energy_saving_percent, estimate_round_energy
from ecotrust.io import read_json, write_json


STAGE = 13


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", "", "nan", "none"}:
        return False
    raise ValueError(f"Cannot parse boolean telemetry value: {value!r}")


def _budget_count(total: int, fraction: float) -> int:
    if total < 1:
        return 0
    return max(1, min(total, int(round(total * fraction))))


def _top_budget(frame, scores, budget_fraction: float):
    import numpy as np

    count = _budget_count(len(frame), budget_fraction)
    selected = np.zeros(len(frame), dtype=bool)
    if count:
        order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
        selected[order[:count]] = True
    return selected


def _policy_predictions(frame, policy: str, budget_fraction: float, seed: int, learned_model=None):
    import numpy as np

    predictions = np.zeros(len(frame), dtype=bool)
    for round_id, indices in frame.groupby("round", sort=True).groups.items():
        group = frame.loc[indices]
        if policy == "random_budget":
            rng = np.random.default_rng(seed + int(round_id))
            scores = rng.random(len(group))
        elif policy == "round_robin":
            ordered = np.argsort(group["client_id"].astype(str).to_numpy(), kind="stable")
            count = _budget_count(len(group), budget_fraction)
            scores = np.zeros(len(group), dtype=float)
            if count:
                offset = ((int(round_id) - 1) * count) % len(group)
                chosen = [ordered[(offset + position) % len(group)] for position in range(count)]
                scores[chosen] = 1.0
        elif policy == "static_score":
            scores = group["trust_score"].to_numpy(dtype=float)
        elif policy == "telemetry_heuristic":
            scores = group["priority"].to_numpy(dtype=float)
            eligible = (group["trust_score"].to_numpy(dtype=float) >= 0.55) & (
                group["remaining_energy_j"].to_numpy(dtype=float) >= 120.0
            )
            scores = np.where(eligible, scores, -1e9)
        elif policy == "learned_policy":
            if learned_model is None:
                raise ValueError("learned_model is required for learned_policy")
            scores = learned_model.predict_proba(group[list(LEARNED_FEATURES)])[:, 1]
        else:
            raise ValueError(f"Unknown stage-13 policy: {policy}")
        predictions[frame.index.get_indexer(indices)] = _top_budget(group, scores, budget_fraction)
    return predictions


LEARNED_FEATURES = (
    "trust_score",
    "drift_score",
    "remaining_energy_fraction",
    "round_fraction",
)


def _load_telemetry(cfg, dataset: str, run_mode: str):
    import pandas as pd

    history_path = cfg.results_root / dataset / "agentic" / run_mode / "federated_round_history.csv"
    summary_path = cfg.results_root / dataset / "agentic" / run_mode / "federated_summary.json"
    splits_path = cfg.results_root / dataset / "preprocessed" / run_mode / "client_splits.json"
    if not history_path.is_file() or not summary_path.is_file() or not splits_path.is_file():
        raise FileNotFoundError(f"{dataset}: missing stage-06 telemetry prerequisites")
    frame = pd.read_csv(history_path)
    required = {"round", "client_id", "selected", "priority", "trust_score", "drift_score", "remaining_energy_j"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{dataset}: telemetry is missing columns {sorted(missing)}")
    split_records = read_json(splits_path)
    initial_energy = {
        str(item["client_id"]): max(float(item["energy_budget_j"]), 1e-12)
        for item in split_records
    }
    denominators = frame["client_id"].astype(str).map(initial_energy)
    if denominators.isna().any():
        missing_clients = sorted(frame.loc[denominators.isna(), "client_id"].astype(str).unique())
        raise ValueError(f"{dataset}: telemetry clients missing initial energy budgets: {missing_clients}")
    frame["remaining_energy_fraction"] = (frame["remaining_energy_j"] / denominators).clip(0.0, 1.0)
    frame["round_fraction"] = frame["round"] / max(float(frame["round"].max()), 1.0)
    frame["selected"] = frame["selected"].map(_as_bool)
    frame["dataset"] = dataset
    summary = read_json(summary_path)
    split_sizes = {item["client_id"]: len(item["indices"]) for item in split_records}
    return frame, summary, split_sizes


def run_stage13(
    config_path: str | Path,
    registry_path: str | Path,
    *,
    run_mode: str = "full",
    datasets: Iterable[str] | None = None,
) -> dict:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = load_config(config_path)
    registry = read_json(Path(registry_path))
    stage_cfg = registry["stages"]["13"]
    selected_datasets = list(datasets or registry["data"]["datasets"])
    budget_fraction = float(stage_cfg["budget_fraction"])
    seed = int(stage_cfg["seed"])
    output_root = cfg.results_root / "publication" / "stage13"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.logs_root / f"13_cross_dataset_controller_generalization_{run_mode}_manifest.json"
    manifest = {
        "stage": STAGE,
        "status": "running",
        "mode": run_mode,
        "started_at": _utc_now(),
        "folds": {},
    }
    write_json(manifest_path, manifest)
    try:
        telemetry = {}
        summaries = {}
        split_sizes = {}
        for dataset in selected_datasets:
            telemetry[dataset], summaries[dataset], split_sizes[dataset] = _load_telemetry(cfg, dataset, run_mode)
        prediction_rows = []
        metric_rows = []
        for held_out in selected_datasets:
            training = pd.concat(
                [telemetry[dataset] for dataset in selected_datasets if dataset != held_out],
                ignore_index=True,
            )
            testing = telemetry[held_out].reset_index(drop=True).copy()
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            )
            model.fit(training[list(LEARNED_FEATURES)], training["selected"].astype(int))
            reference = testing["selected"].to_numpy(dtype=bool)
            model_size_mb = float(summaries[held_out]["model_size_mb"])
            all_energy = 0.0
            for client_id in testing["client_id"]:
                all_energy += estimate_round_energy(split_sizes[held_out][str(client_id)], model_size_mb, True)["total_j"]
            for policy in stage_cfg["policies"]:
                predicted = _policy_predictions(
                    testing,
                    policy,
                    budget_fraction,
                    seed + selected_datasets.index(held_out) * 1000,
                    learned_model=model,
                )
                policy_energy = 0.0
                for client_id, selected in zip(testing["client_id"], predicted):
                    policy_energy += estimate_round_energy(
                        split_sizes[held_out][str(client_id)],
                        model_size_mb,
                        bool(selected),
                    )["total_j"]
                metric_rows.append(
                    {
                        "held_out_dataset": held_out,
                        "training_datasets": ";".join(dataset for dataset in selected_datasets if dataset != held_out),
                        "policy": policy,
                        "decision_accuracy": float(accuracy_score(reference, predicted)),
                        "decision_balanced_accuracy": float(balanced_accuracy_score(reference, predicted)),
                        "decision_precision": float(precision_score(reference, predicted, zero_division=0)),
                        "decision_recall": float(recall_score(reference, predicted, zero_division=0)),
                        "decision_f1": float(f1_score(reference, predicted, zero_division=0)),
                        "selected_fraction": float(np.mean(predicted)),
                        "estimated_energy_j": float(policy_energy),
                        "estimated_all_client_energy_j": float(all_energy),
                        "estimated_energy_saving_percent": energy_saving_percent(all_energy, policy_energy),
                        "energy_measurement_type": "model_estimate",
                    }
                )
                for row_id, (client_id, round_id, truth, decision) in enumerate(
                    zip(testing["client_id"], testing["round"], reference, predicted)
                ):
                    prediction_rows.append(
                        {
                            "held_out_dataset": held_out,
                            "policy": policy,
                            "row_id": row_id,
                            "round": int(round_id),
                            "client_id": str(client_id),
                            "reference_selected": bool(truth),
                            "policy_selected": bool(decision),
                        }
                    )
            manifest["folds"][held_out] = {
                "status": "completed",
                "training_datasets": [dataset for dataset in selected_datasets if dataset != held_out],
                "test_rows": int(len(testing)),
                "refit_on_held_out": False,
            }
            write_json(manifest_path, manifest)
        metrics_path = output_root / "controller_generalization_metrics.csv"
        predictions_path = output_root / "controller_predictions.csv"
        pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
        pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
        manifest.update(
            {
                "status": "completed",
                "completed_at": _utc_now(),
                "artifacts": {
                    "metrics": str(metrics_path),
                    "predictions": str(predictions_path),
                },
            }
        )
        write_json(manifest_path, manifest)
        return {"manifest": str(manifest_path), **manifest}
    except BaseException as exc:
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
