from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import time
import traceback
from typing import Iterable

from ecotrust.agent.controller import AgentDecision, ClientState, EcoTrustAgent
from ecotrust.config import load_config, run_mode_settings
from ecotrust.data.clients import ClientSplit
from ecotrust.data.preprocess import load_preprocessed
from ecotrust.energy.simulator import energy_saving_percent, estimate_round_energy
from ecotrust.federated.client import train_local_model
from ecotrust.federated.server import weighted_average_state_dicts
from ecotrust.io import read_json, write_json
from ecotrust.models.embedding import load_embedding_model


STAGE = 9
METHODS = (
    "fedavg",
    "fedprox",
    "fedadam",
    "trust_weighted_all_clients",
    "ecotrust_agent",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_checkpoint(torch, path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _predict_metrics(model, X, y, *, device: str, batch_size: int) -> dict:
    import numpy as np
    import torch
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        matthews_corrcoef,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.metrics import auc

    X_array = np.asarray(X)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_array), batch_size):
            xb = torch.tensor(X_array[start : start + batch_size], dtype=torch.float32).to(device)
            output = model(xb)
            logits = output[0] if isinstance(output, tuple) else output
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    score = np.concatenate(probabilities) if probabilities else np.empty(0, dtype=float)
    if not np.isfinite(score).all():
        non_finite = int((~np.isfinite(score)).sum())
        raise FloatingPointError(
            f"Model produced {non_finite} non-finite probability values; "
            "the run is invalid and must not be scored."
        )
    pred = (score >= 0.5).astype(int)
    y_array = np.asarray(y, dtype=int)
    precision_curve, recall_curve, _ = precision_recall_curve(y_array, score)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (score >= left) & (score < right if right < 1.0 else score <= right)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y_array[mask].mean()) - float(score[mask].mean()))
    return {
        "accuracy": float(accuracy_score(y_array, pred)),
        "precision": float(precision_score(y_array, pred, zero_division=0)),
        "recall": float(recall_score(y_array, pred, zero_division=0)),
        "attack_f1": float(f1_score(y_array, pred, zero_division=0)),
        "macro_f1": float(f1_score(y_array, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_array, pred)),
        "mcc": float(matthews_corrcoef(y_array, pred)),
        "roc_auc": float(roc_auc_score(y_array, score)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
        "brier_score": float(brier_score_loss(y_array, score)),
        "expected_calibration_error": ece,
    }


def _method_settings(method: str) -> dict:
    if method not in METHODS:
        raise ValueError(f"Unknown stage-09 method: {method}")
    return {
        "selection": "agent" if method == "ecotrust_agent" else "all",
        "trust_weighted": method in {"trust_weighted_all_clients", "ecotrust_agent"},
        "proximal_mu": 0.01 if method == "fedprox" else 0.0,
        "server_optimizer": "fedadam" if method == "fedadam" else "average",
    }


def _server_update(
    torch,
    global_state: dict,
    averaged_state: dict,
    optimizer_state: dict,
    method: str,
    fedadam_config: dict | None = None,
):
    if method != "fedadam":
        return averaged_state, optimizer_state
    config = fedadam_config or {}
    beta1 = float(config.get("beta1", 0.9))
    beta2 = float(config.get("beta2", 0.99))
    eta = float(config.get("server_learning_rate", 0.01))
    tau = float(config.get("tau", 1e-3))
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0 and eta > 0.0 and tau > 0.0):
        raise ValueError("Invalid FedAdam server optimizer configuration")
    step = int(optimizer_state.get("step", 0)) + 1
    first_moment = optimizer_state.setdefault("first_moment", {})
    second_moment = optimizer_state.setdefault("second_moment", {})
    updated = deepcopy(global_state)
    for key, current in global_state.items():
        if not hasattr(current, "is_floating_point") or not current.is_floating_point():
            updated[key] = current
            continue
        delta = averaged_state[key] - current
        m = first_moment.get(key, torch.zeros_like(delta))
        v = second_moment.get(key, torch.zeros_like(delta))
        m = beta1 * m + (1.0 - beta1) * delta
        v = beta2 * v + (1.0 - beta2) * delta.square()
        first_moment[key] = m
        second_moment[key] = v
        updated[key] = current + eta * m / (torch.sqrt(v) + tau)
        if not torch.isfinite(updated[key]).all():
            raise FloatingPointError(f"FedAdam produced a non-finite tensor for parameter {key!r}")
    optimizer_state["step"] = step
    optimizer_state["config"] = {
        "beta1": beta1,
        "beta2": beta2,
        "server_learning_rate": eta,
        "tau": tau,
    }
    return updated, optimizer_state


def run_single_benchmark(
    *,
    config_path: str | Path,
    dataset: str,
    method: str,
    seed: int,
    run_mode: str = "full",
    device: str = "cpu",
    resume: bool = True,
    budget_fraction: float = 0.6,
    fedadam_config: dict | None = None,
) -> dict:
    import numpy as np
    import pandas as pd
    import torch

    cfg = load_config(config_path)
    settings = run_mode_settings(cfg, run_mode)
    rounds = int(settings["rounds"])
    local_epochs = int(settings["local_epochs"])
    batch_size = 512 if device.startswith("cuda") else 256
    method_cfg = _method_settings(method)
    prep = cfg.results_root / dataset / "preprocessed" / run_mode
    embedding_checkpoint = cfg.results_root / dataset / "embedding" / run_mode / "ecotrust_encoder.pt"
    output_dir = cfg.results_root / "publication" / "stage09" / dataset / method / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    progress_path = output_dir / "progress.json"
    rounds_path = output_dir / "round_metrics.csv"
    summary_path = output_dir / "summary.json"
    model_path = output_dir / "global_model.pt"
    if resume and summary_path.is_file():
        prior = read_json(summary_path)
        if prior.get("status") == "completed" and int(prior.get("completed_rounds", 0)) == rounds:
            return {**prior, "skipped": True}

    X_train, X_test, y_train, y_test = load_preprocessed(prep)
    splits = [
        ClientSplit(**item)
        for item in json.loads((prep / "client_splits.json").read_text(encoding="utf-8"))
    ]
    model, _ = load_embedding_model(embedding_checkpoint, device)
    global_model = deepcopy(model).to(device)
    model_bytes = int(sum(parameter.numel() * parameter.element_size() for parameter in global_model.parameters()))
    model_size_mb = model_bytes / (1024**2)
    telemetry = {
        split.client_id: {"recent_loss": 0.5, "recent_f1": 0.5, "drift_score": 0.0}
        for split in splits
    }
    energy_remaining = {split.client_id: float(split.energy_budget_j) for split in splits}
    agent = EcoTrustAgent()
    round_rows: list[dict] = []
    completed_round = 0
    optimizer_state: dict = {}
    elapsed_before = 0.0
    started_at = _utc_now()
    if resume and checkpoint_path.is_file():
        checkpoint = _load_checkpoint(torch, checkpoint_path, device)
        if checkpoint.get("method") != method or int(checkpoint.get("seed", -1)) != seed:
            raise RuntimeError(f"Checkpoint protocol mismatch at {checkpoint_path}")
        global_model.load_state_dict(checkpoint["model_state"])
        completed_round = int(checkpoint["completed_round"])
        round_rows = list(checkpoint.get("round_rows", []))
        telemetry.update(checkpoint.get("telemetry", {}))
        energy_remaining.update(checkpoint.get("energy_remaining", {}))
        optimizer_state = checkpoint.get("optimizer_state", {})
        if method == "fedadam" and optimizer_state.get("config") != fedadam_config:
            raise RuntimeError(
                "FedAdam checkpoint optimizer configuration does not match the registered protocol"
            )
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        started_at = str(checkpoint.get("started_at", started_at))

    invocation_started = time.perf_counter()
    cpu_started = time.process_time()
    rng = np.random.default_rng(seed)
    try:
        for round_id in range(completed_round + 1, rounds + 1):
            if method_cfg["selection"] == "agent":
                decisions = agent.decide(
                    [
                        ClientState(
                            client_id=split.client_id,
                            trust_score=split.trust_score,
                            energy_budget_j=energy_remaining[split.client_id],
                            recent_loss=telemetry[split.client_id]["recent_loss"],
                            recent_f1=telemetry[split.client_id]["recent_f1"],
                            drift_score=telemetry[split.client_id]["drift_score"],
                        )
                        for split in splits
                    ],
                    budget_fraction=budget_fraction,
                )
            else:
                decisions = [AgentDecision(split.client_id, True, "all", 1.0, 0) for split in splits]
            selected = {decision.client_id for decision in decisions if decision.selected}
            if not selected:
                selected = {splits[int(rng.integers(0, len(splits)))].client_id}
            global_state = deepcopy(global_model.state_dict())
            client_states: list[tuple[dict, float]] = []
            round_energy = 0.0
            baseline_energy = 0.0
            selected_count = 0
            for client_number, split in enumerate(splits):
                is_selected = split.client_id in selected
                estimated = estimate_round_energy(len(split.indices), model_size_mb, is_selected)
                all_client_estimated = estimate_round_energy(len(split.indices), model_size_mb, True)
                round_energy += estimated["total_j"]
                baseline_energy += all_client_estimated["total_j"]
                energy_remaining[split.client_id] = max(
                    0.0, energy_remaining[split.client_id] - estimated["total_j"]
                )
                if not is_selected:
                    continue
                selected_count += 1
                local_model = deepcopy(global_model)
                state, local_metrics = train_local_model(
                    local_model,
                    X_train.iloc[split.indices],
                    np.asarray(y_train)[split.indices],
                    epochs=local_epochs,
                    batch_size=batch_size,
                    device=device,
                    seed=seed + round_id * 1000 + client_number,
                    global_state=global_state,
                    proximal_mu=float(method_cfg["proximal_mu"]),
                )
                weight = len(split.indices)
                if method_cfg["trust_weighted"]:
                    weight *= max(float(split.trust_score), 0.01)
                client_states.append((state, float(weight)))
                telemetry[split.client_id]["recent_loss"] = float(local_metrics["loss"])
                telemetry[split.client_id]["recent_f1"] = float(local_metrics["f1"])
            averaged = weighted_average_state_dicts(client_states)
            updated, optimizer_state = _server_update(
                torch,
                global_state,
                averaged,
                optimizer_state,
                method,
                fedadam_config=fedadam_config,
            )
            global_model.load_state_dict(updated)
            metrics = _predict_metrics(
                global_model, X_test, y_test, device=device, batch_size=max(batch_size, 2048)
            )
            row = {
                "round": round_id,
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "selected_clients": selected_count,
                "total_clients": len(splits),
                "estimated_round_energy_j": round_energy,
                "estimated_all_client_energy_j": baseline_energy,
                "estimated_energy_saving_percent": energy_saving_percent(baseline_energy, round_energy),
                "communication_bytes": int(selected_count * model_bytes * 2),
                **metrics,
            }
            round_rows.append(row)
            completed_round = round_id
            pd.DataFrame(round_rows).to_csv(rounds_path, index=False)
            checkpoint = {
                "method": method,
                "seed": seed,
                "started_at": started_at,
                "completed_round": completed_round,
                "rounds_requested": rounds,
                "model_state": global_model.state_dict(),
                "optimizer_state": optimizer_state,
                "round_rows": round_rows,
                "telemetry": telemetry,
                "energy_remaining": energy_remaining,
                "elapsed_seconds": elapsed_before + time.perf_counter() - invocation_started,
            }
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)
            _atomic_json(
                progress_path,
                {
                    "status": "running",
                    "method": method,
                    "seed": seed,
                    "completed_round": completed_round,
                    "rounds_requested": rounds,
                    "updated_at": _utc_now(),
                },
            )
            print(
                f"{dataset} {method} seed={seed} round={round_id}/{rounds} "
                f"macro_f1={metrics['macro_f1']:.6f}",
                flush=True,
            )
    except BaseException as exc:
        _atomic_json(
            progress_path,
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "method": method,
                "seed": seed,
                "completed_round": completed_round,
                "rounds_requested": rounds,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "updated_at": _utc_now(),
            },
        )
        raise

    torch.save(global_model.state_dict(), model_path)
    elapsed = elapsed_before + time.perf_counter() - invocation_started
    summary = {
        "status": "completed",
        "stage": STAGE,
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "device": device,
        "completed_rounds": completed_round,
        "rounds_requested": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "budget_fraction": budget_fraction,
        "fedadam_config": fedadam_config if method == "fedadam" else None,
        "final_metrics": round_rows[-1],
        "estimated_energy_j": float(sum(row["estimated_round_energy_j"] for row in round_rows)),
        "estimated_all_client_energy_j": float(
            sum(row["estimated_all_client_energy_j"] for row in round_rows)
        ),
        "communication_bytes": int(sum(row["communication_bytes"] for row in round_rows)),
        "wall_time_seconds": float(elapsed),
        "process_cpu_seconds_this_invocation": float(time.process_time() - cpu_started),
        "peak_rss_mb": _peak_rss_mb(),
        "energy_measurement_type": "model_estimate",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "round_metrics": str(rounds_path),
            "global_model": str(model_path),
            "progress": str(progress_path),
        },
    }
    _atomic_json(summary_path, summary)
    _atomic_json(progress_path, {**summary, "status": "completed"})
    return summary


def run_stage09(
    config_path: str | Path,
    registry_path: str | Path,
    *,
    run_mode: str = "full",
    device: str = "cpu",
    datasets: Iterable[str] | None = None,
    resume: bool = True,
) -> dict:
    cfg = load_config(config_path)
    registry = read_json(Path(registry_path))
    if registry.get("status") != "frozen_before_stage_09":
        raise RuntimeError("Publication registry must be frozen before stage 09")
    selected_datasets = list(datasets or registry["data"]["datasets"])
    methods = list(registry["stages"]["09"]["methods"])
    fedadam_config = dict(registry["stages"]["09"]["fedadam"])
    seeds = [int(seed) for seed in registry["randomness"]["primary_seeds"]]
    manifest_path = cfg.logs_root / f"09_recent_baselines_and_multiseed_{run_mode}_manifest.json"
    manifest = {
        "stage": STAGE,
        "status": "running",
        "mode": run_mode,
        "device": device,
        "registry": str(registry_path),
        "started_at": _utc_now(),
        "runs": {},
    }
    if resume and manifest_path.is_file():
        previous = read_json(manifest_path)
        if previous.get("status") in {"running", "failed", "interrupted"}:
            manifest["started_at"] = previous.get("started_at", manifest["started_at"])
            manifest["runs"] = previous.get("runs", {})
    write_json(manifest_path, manifest)
    try:
        for dataset in selected_datasets:
            for method in methods:
                for seed in seeds:
                    key = f"{dataset}/{method}/seed_{seed}"
                    manifest["runs"][key] = {"status": "running", "updated_at": _utc_now()}
                    write_json(manifest_path, manifest)
                    result = run_single_benchmark(
                        config_path=config_path,
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        run_mode=run_mode,
                        device=device,
                        resume=resume,
                        budget_fraction=float(registry["training"]["client_fraction"]),
                        fedadam_config=fedadam_config,
                    )
                    manifest["runs"][key] = {
                        "status": "completed",
                        "skipped": bool(result.get("skipped", False)),
                        "summary": result,
                    }
                    write_json(manifest_path, manifest)
        manifest.update({"status": "completed", "completed_at": _utc_now()})
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
