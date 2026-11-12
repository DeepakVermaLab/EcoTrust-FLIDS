from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Iterable

from ecotrust.agent.controller import EcoTrustAgent
from ecotrust.config import load_config, run_mode_settings
from ecotrust.data.clients import ClientSplit
from ecotrust.data.preprocess import load_preprocessed
from ecotrust.federated.incremental import run_incremental_federated_training
from ecotrust.io import read_json, write_json
from ecotrust.models.embedding import load_embedding_model


ABLATION_VARIANTS = ("no_trust", "no_energy", "no_drift")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_state(torch, path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _agent_for_variant(variant: str) -> EcoTrustAgent:
    if variant == "no_trust":
        return EcoTrustAgent(min_trust=0.0, trust_weight=0.0)
    if variant == "no_energy":
        return EcoTrustAgent(min_energy_j=0.0, energy_weight=0.0)
    if variant == "no_drift":
        return EcoTrustAgent(drift_boost=0.0)
    raise KeyError(f"Unknown ablation variant: {variant}")


def _splits_for_variant(splits: list[ClientSplit], variant: str) -> list[ClientSplit]:
    if variant != "no_trust":
        return splits
    return [replace(split, trust_score=1.0) for split in splits]


def _validate_completed(summary: dict, dataset: str, name: str, rounds: int) -> None:
    if summary.get("status") != "completed":
        raise RuntimeError(f"{dataset}: {name} is not completed")
    if int(summary.get("completed_rounds", 0)) != rounds:
        raise RuntimeError(
            f"{dataset}: {name} has {summary.get('completed_rounds')} rounds; expected {rounds}"
        )


def _ece(y_true, probabilities, bins: int = 10) -> float:
    import numpy as np

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    confidence = np.maximum(p, 1.0 - p)
    correctness = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(y), 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if mask.any():
            value += float(mask.sum()) / total * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    return value


def _predict_probabilities(model, X, device: str, batch_size: int = 4096):
    import numpy as np
    import torch

    values = np.asarray(X)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            xb = torch.tensor(values[start : start + batch_size], dtype=torch.float32).to(device)
            logits = model(xb)
            logits = logits[0] if isinstance(logits, tuple) else logits
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty(0, dtype=float)


def _probability_metrics(y_true, probabilities: object) -> dict:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= 0.5).astype(int)
    both_classes = len(np.unique(y)) == 2
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else None,
        "average_precision": float(average_precision_score(y, p)) if both_classes else None,
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": _ece(y, p, bins=10),
    }


def _prior_shift_indices(y, attack_prior: float, seed: int):
    import numpy as np

    labels = np.asarray(y, dtype=int)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Controlled prior-shift evaluation requires both classes")
    if not 0.0 < attack_prior < 1.0:
        raise ValueError("attack_prior must be in (0, 1)")
    maximum = int(min(len(positive) / attack_prior, len(negative) / (1.0 - attack_prior)))
    positive_count = max(1, min(len(positive), int(round(maximum * attack_prior))))
    negative_count = max(1, min(len(negative), maximum - positive_count))
    rng = np.random.default_rng(seed)
    chosen = np.concatenate(
        [rng.choice(negative, negative_count, replace=False), rng.choice(positive, positive_count, replace=False)]
    )
    rng.shuffle(chosen)
    return chosen


def _evaluate_prior_shift(
    dataset: str,
    model_name: str,
    model,
    X_test,
    y_test,
    *,
    device: str,
    seed: int,
) -> list[dict]:
    import numpy as np

    rows = []
    natural_probabilities = _predict_probabilities(model, X_test, device)
    natural = _probability_metrics(y_test, natural_probabilities)
    rows.append(
        {
            "dataset": dataset,
            "model": model_name,
            "scenario": "natural_test_prior",
            "target_attack_prior": None,
            "observed_attack_prior": float(np.asarray(y_test).mean()),
            "samples": int(len(y_test)),
            "f1_retention_vs_natural": 1.0,
            **natural,
        }
    )
    for offset, prior in enumerate((0.2, 0.5, 0.8)):
        indices = _prior_shift_indices(y_test, prior, seed + offset)
        y_subset = np.asarray(y_test)[indices]
        X_subset = X_test.iloc[indices] if hasattr(X_test, "iloc") else np.asarray(X_test)[indices]
        metrics = _probability_metrics(y_subset, _predict_probabilities(model, X_subset, device))
        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "scenario": "controlled_label_prior_shift",
                "target_attack_prior": prior,
                "observed_attack_prior": float(y_subset.mean()),
                "samples": int(len(indices)),
                "f1_retention_vs_natural": (
                    float(metrics["f1"]) / float(natural["f1"]) if natural["f1"] > 0 else None
                ),
                **metrics,
            }
        )
    return rows


def run_ablation_and_drift_analysis(
    config_path: str | Path,
    *,
    run_mode: str = "full",
    datasets: Iterable[str] = ("Edge-IIoTset", "TON_IoT", "CICIoT2023"),
    device: str = "cpu",
    budget_fraction: float = 0.6,
    batch_size: int | None = None,
    resume: bool = True,
) -> dict:
    """Run component ablations and controlled test-prior drift evaluation.

    The drift protocol changes the held-out label prior only; it is explicitly
    not presented as natural temporal drift or continual-learning forgetting.
    """

    import pandas as pd
    import torch

    cfg = load_config(config_path)
    settings = run_mode_settings(cfg, run_mode)
    rounds = int(settings["rounds"])
    local_epochs = int(settings["local_epochs"])
    effective_batch_size = batch_size or (512 if device.startswith("cuda") else 256)
    output_root = cfg.results_root / "ablation_drift" / run_mode
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.logs_root / f"07_ablation_and_drift_analysis_{run_mode}_manifest.json"
    log_path = cfg.logs_root / f"07_ablation_and_drift_analysis_{run_mode}.log"
    manifest = {
        "stage": 7,
        "status": "running",
        "mode": run_mode,
        "device": device,
        "budget_fraction": budget_fraction,
        "protocol": "matched component ablations plus controlled held-out label-prior shift",
        "drift_scope_note": "Controlled label-prior robustness; not natural temporal drift or forgetting.",
        "started_at": _utc_now(),
        "datasets": {},
    }
    write_json(manifest_path, manifest)
    summary_rows: list[dict] = []
    drift_rows: list[dict] = []

    for dataset_index, dataset in enumerate(datasets):
        dataset_state = manifest["datasets"].setdefault(dataset, {"status": "running", "variants": {}})
        write_json(manifest_path, manifest)
        prep = cfg.results_root / dataset / "preprocessed" / run_mode
        embedding_checkpoint = cfg.results_root / dataset / "embedding" / run_mode / "ecotrust_encoder.pt"
        baseline_dir = cfg.results_root / dataset / "federated" / run_mode
        agentic_dir = cfg.results_root / dataset / "agentic" / run_mode
        baseline_summary = read_json(baseline_dir / "federated_summary.json")
        agentic_summary = read_json(agentic_dir / "federated_summary.json")
        _validate_completed(baseline_summary, dataset, "stage-05 all-client baseline", rounds)
        _validate_completed(agentic_summary, dataset, "stage-06 full agentic model", rounds)
        X_train, X_test, y_train, y_test = load_preprocessed(prep)
        splits = [
            ClientSplit(**item)
            for item in json.loads((prep / "client_splits.json").read_text(encoding="utf-8"))
        ]

        for label, summary in (("all_clients", baseline_summary), ("full_agentic", agentic_summary)):
            summary_rows.append(
                {
                    "dataset": dataset,
                    "variant": label,
                    "source_stage": 5 if label == "all_clients" else 6,
                    **summary["final_metrics"],
                    "energy_j": float(summary["proposed_energy_j"]),
                    "energy_saving_percent": float(summary["energy_saving_percent"]),
                    "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0)),
                }
            )

        for variant in ABLATION_VARIANTS:
            variant_dir = cfg.results_root / dataset / "ablation" / variant / run_mode
            summary_path = variant_dir / "federated_summary.json"
            if resume and summary_path.exists():
                result = read_json(summary_path)
                _validate_completed(result, dataset, variant, rounds)
                skipped = True
            else:
                model, _ = load_embedding_model(embedding_checkpoint, device)
                dataset_state["variants"][variant] = {"status": "running", "output_dir": str(variant_dir)}
                write_json(manifest_path, manifest)
                print("Ablation", dataset, variant, flush=True)
                result = run_incremental_federated_training(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    _splits_for_variant(splits, variant),
                    variant_dir,
                    rounds=rounds,
                    local_epochs=local_epochs,
                    batch_size=effective_batch_size,
                    device=device,
                    seed=cfg.seed,
                    resume=resume,
                    log_path=log_path,
                    selection_policy="agent",
                    budget_fraction=budget_fraction,
                    agent=_agent_for_variant(variant),
                )
                skipped = False
            dataset_state["variants"][variant] = {
                "status": "completed",
                "skipped": skipped,
                "output_dir": str(variant_dir),
            }
            write_json(manifest_path, manifest)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "source_stage": 7,
                    **result["final_metrics"],
                    "energy_j": float(result["proposed_energy_j"]),
                    "energy_saving_percent": float(result["energy_saving_percent"]),
                    "elapsed_seconds": float(result.get("elapsed_seconds", 0.0)),
                }
            )

        for model_name, source_dir in (("all_clients", baseline_dir), ("full_agentic", agentic_dir)):
            model, _ = load_embedding_model(embedding_checkpoint, device)
            model.load_state_dict(_load_state(torch, source_dir / "federated_global_model.pt", device))
            drift_rows.extend(
                _evaluate_prior_shift(
                    dataset,
                    model_name,
                    model,
                    X_test,
                    y_test,
                    device=device,
                    seed=cfg.seed + dataset_index * 100,
                )
            )

        dataset_state["status"] = "completed"
        write_json(manifest_path, manifest)

    summary = pd.DataFrame(summary_rows)
    full = summary[summary["variant"] == "full_agentic"].set_index("dataset")
    summary["f1_delta_vs_full_agentic"] = summary.apply(
        lambda row: float(row["f1"]) - float(full.loc[row["dataset"], "f1"]), axis=1
    )
    summary["energy_delta_percent_vs_full_agentic"] = summary.apply(
        lambda row: 100.0
        * (float(row["energy_j"]) - float(full.loc[row["dataset"], "energy_j"]))
        / max(float(full.loc[row["dataset"], "energy_j"]), 1e-12),
        axis=1,
    )
    summary_path = output_root / "ablation_summary.csv"
    drift_path = output_root / "controlled_prior_drift_metrics.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(drift_rows).to_csv(drift_path, index=False)
    manifest.update(
        {
            "status": "completed",
            "completed_at": _utc_now(),
            "artifacts": {"ablation_summary": str(summary_path), "prior_drift_metrics": str(drift_path)},
        }
    )
    write_json(manifest_path, manifest)
    return {"manifest": str(manifest_path), **manifest}


def run_with_failure_manifest(*args, **kwargs) -> dict:
    """Preserve a stage-level traceback when setup fails before a dataset run."""
    config_path = kwargs.get("config_path") or (args[0] if args else None)
    run_mode = kwargs.get("run_mode", "full")
    try:
        return run_ablation_and_drift_analysis(*args, **kwargs)
    except BaseException as exc:
        if config_path is not None:
            cfg = load_config(config_path)
            path = cfg.logs_root / f"07_ablation_and_drift_analysis_{run_mode}_manifest.json"
            prior = read_json(path) if path.exists() else {"stage": 7, "mode": run_mode}
            prior.update(
                {
                    "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    "failed_at": _utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            write_json(path, prior)
        raise
