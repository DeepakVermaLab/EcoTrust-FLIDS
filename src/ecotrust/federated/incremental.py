from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from ecotrust.agent.controller import AgentDecision, ClientState, EcoTrustAgent
from ecotrust.energy.simulator import energy_saving_percent, estimate_round_energy
from ecotrust.federated.client import train_local_model
from ecotrust.federated.server import evaluate_torch_model, weighted_average_state_dicts


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def _load_torch_checkpoint(torch, path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_incremental_federated_training(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    splits,
    output_dir: str | Path,
    rounds: int = 3,
    local_epochs: int = 1,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    agent: EcoTrustAgent | None = None,
    model_size_mb: float | None = None,
    budget_fraction: float = 0.6,
    seed: int = 42,
    resume: bool = True,
    log_path: str | Path | None = None,
    selection_policy: str = "agent",
) -> dict:
    """Run trust-weighted FL with deterministic telemetry and round-level resume.

    The latest global model, client history, round metrics, energy state, and
    telemetry are persisted after each completed round. An interrupted run can
    continue from the next round without repeating completed work.
    """
    try:
        import numpy as np
        import pandas as pd
        import torch
    except ImportError as exc:
        raise ImportError("Install project dependencies first: python -m pip install -r requirements.txt") from exc

    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    if selection_policy not in {"agent", "all"}:
        raise ValueError("selection_policy must be 'agent' or 'all'")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "federated_checkpoint.pt"
    progress_path = out / "federated_progress.json"
    history_path = out / "federated_round_history.csv"
    metrics_path = out / "federated_round_metrics.csv"
    final_model_path = out / "federated_global_model.pt"
    summary_path = out / "federated_summary.json"
    persistent_log = Path(log_path) if log_path else out / "federated_training.log"
    persistent_log.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"[{_utc_now()}] {message}"
        print(line, flush=True)
        with persistent_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    agent = agent or EcoTrustAgent()
    global_model = deepcopy(model).to(device)
    labels = np.asarray(y_train)
    global_attack_rate = float(labels.mean()) if len(labels) else 0.0
    model_size_mb = model_size_mb or sum(
        parameter.numel() * parameter.element_size() for parameter in global_model.parameters()
    ) / (1024**2)

    telemetry = {}
    energy_remaining = {}
    for split in splits:
        client_labels = labels[split.indices]
        client_attack_rate = float(client_labels.mean()) if len(client_labels) else global_attack_rate
        telemetry[split.client_id] = {
            "recent_loss": 0.5,
            "recent_f1": 0.5,
            "drift_score": min(abs(client_attack_rate - global_attack_rate) * 2.0, 1.0),
        }
        energy_remaining[split.client_id] = float(split.energy_budget_j)

    history: list[dict] = []
    round_metrics: list[dict] = []
    completed_round = 0
    started_at = _utc_now()
    elapsed_before_resume = 0.0

    if resume and checkpoint_path.exists():
        checkpoint = _load_torch_checkpoint(torch, checkpoint_path, device)
        if int(checkpoint.get("rounds_requested", rounds)) != rounds:
            raise ValueError(
                f"Existing checkpoint requested {checkpoint.get('rounds_requested')} rounds, "
                f"but this run requests {rounds}. Use a clean output directory for a different protocol."
            )
        checkpoint_policy = str(checkpoint.get("selection_policy", selection_policy))
        if checkpoint_policy != selection_policy:
            raise ValueError(
                f"Existing checkpoint uses selection_policy={checkpoint_policy!r}, "
                f"but this run requests {selection_policy!r}."
            )
        global_model.load_state_dict(checkpoint["model_state"])
        completed_round = int(checkpoint.get("completed_round", 0))
        history = list(checkpoint.get("history", []))
        round_metrics = list(checkpoint.get("round_metrics", []))
        telemetry.update(checkpoint.get("telemetry", {}))
        energy_remaining.update(checkpoint.get("energy_remaining", {}))
        started_at = str(checkpoint.get("started_at", started_at))
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))
        log(f"Resuming from completed round {completed_round}/{rounds} on {device}.")
    else:
        log(f"Starting {rounds} federated rounds on {device}; model_size_mb={model_size_mb:.4f}.")

    invocation_started = time.perf_counter()

    def elapsed_seconds() -> float:
        return elapsed_before_resume + (time.perf_counter() - invocation_started)

    def write_progress(status: str, error: str | None = None) -> None:
        payload = {
            "status": status,
            "started_at": started_at,
            "updated_at": _utc_now(),
            "completed_round": completed_round,
            "rounds_requested": rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "device": device,
            "seed": seed,
            "budget_fraction": budget_fraction,
            "selection_policy": selection_policy,
            "model_size_mb": model_size_mb,
            "elapsed_seconds": elapsed_seconds(),
        }
        if round_metrics:
            payload["latest_metrics"] = round_metrics[-1]
        if error:
            payload["error"] = error
        _write_json(progress_path, payload)

    write_progress("running")
    try:
        for round_id in range(completed_round + 1, rounds + 1):
            states = [
                ClientState(
                    client_id=split.client_id,
                    trust_score=split.trust_score,
                    energy_budget_j=energy_remaining[split.client_id],
                    recent_loss=telemetry[split.client_id]["recent_loss"],
                    recent_f1=telemetry[split.client_id]["recent_f1"],
                    drift_score=telemetry[split.client_id]["drift_score"],
                )
                for split in splits
            ]
            if selection_policy == "all":
                decisions = [
                    AgentDecision(state.client_id, True, "all_clients_baseline", 1.0, 0)
                    for state in states
                ]
            else:
                decisions = agent.decide(states, budget_fraction=budget_fraction)
            decision_map = {decision.client_id: decision for decision in decisions}
            client_states = []
            current_rows = []
            round_energy = 0.0
            baseline_round_energy = 0.0

            for client_number, split in enumerate(splits):
                decision = decision_map.get(
                    split.client_id,
                    AgentDecision(split.client_id, False, "not_scored", 0.0, 1),
                )
                baseline_energy = estimate_round_energy(len(split.indices), model_size_mb, True)
                energy = estimate_round_energy(len(split.indices), model_size_mb, decision.selected)
                baseline_round_energy += baseline_energy["total_j"]
                round_energy += energy["total_j"]
                energy_remaining[split.client_id] = max(
                    0.0, energy_remaining[split.client_id] - energy["total_j"]
                )
                row = {
                    "round": round_id,
                    "client_id": split.client_id,
                    "selected": decision.selected,
                    "reason": decision.reason,
                    "priority": decision.priority,
                    "sleep_rounds": decision.sleep_rounds,
                    "trust_score": split.trust_score,
                    "drift_score": telemetry[split.client_id]["drift_score"],
                    "remaining_energy_j": energy_remaining[split.client_id],
                    **energy,
                }
                if decision.selected:
                    local_model = deepcopy(global_model)
                    state, metrics = train_local_model(
                        local_model,
                        X_train.iloc[split.indices] if hasattr(X_train, "iloc") else X_train[split.indices],
                        labels[split.indices],
                        epochs=local_epochs,
                        batch_size=batch_size,
                        lr=lr,
                        device=device,
                        seed=seed + round_id * 1000 + client_number,
                    )
                    client_states.append((state, len(split.indices) * max(split.trust_score, 0.01)))
                    telemetry[split.client_id]["recent_loss"] = float(metrics["loss"])
                    telemetry[split.client_id]["recent_f1"] = float(metrics["f1"])
                    row.update({f"local_{key}": value for key, value in metrics.items()})
                    del local_model
                current_rows.append(row)

            if client_states:
                avg_state = weighted_average_state_dicts(client_states)
                global_model.load_state_dict(avg_state)
                del client_states
            eval_metrics = evaluate_torch_model(
                global_model, X_test, y_test, device=device, batch_size=max(batch_size, 1024)
            )
            energy_saving = energy_saving_percent(baseline_round_energy, round_energy)
            for row in current_rows:
                row.update({f"global_{key}": value for key, value in eval_metrics.items()})
                row["round_energy_j"] = round_energy
                row["baseline_round_energy_j"] = baseline_round_energy
                row["energy_saving_percent"] = energy_saving
            history.extend(current_rows)
            round_row = {
                "round": round_id,
                **eval_metrics,
                "selected_clients": sum(row["selected"] for row in current_rows),
                "total_clients": len(current_rows),
                "round_energy_j": round_energy,
                "baseline_round_energy_j": baseline_round_energy,
                "energy_saving_percent": energy_saving,
            }
            round_metrics.append(round_row)
            completed_round = round_id

            pd.DataFrame(history).to_csv(history_path, index=False)
            pd.DataFrame(round_metrics).to_csv(metrics_path, index=False)
            checkpoint = {
                "model_state": global_model.state_dict(),
                "completed_round": completed_round,
                "rounds_requested": rounds,
                "history": history,
                "round_metrics": round_metrics,
                "telemetry": telemetry,
                "energy_remaining": energy_remaining,
                "started_at": started_at,
                "elapsed_seconds": elapsed_seconds(),
                "device": device,
                "seed": seed,
                "selection_policy": selection_policy,
            }
            temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary_checkpoint)
            temporary_checkpoint.replace(checkpoint_path)
            write_progress("running")
            log(
                f"Round {round_id}/{rounds} complete: f1={eval_metrics['f1']:.6f}, "
                f"selected={round_row['selected_clients']}/{round_row['total_clients']}, "
                f"energy_saving={energy_saving:.2f}%."
            )
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        write_progress(status, error)
        log(f"Run {status} after round {completed_round}: {error}")
        raise

    temporary_model = final_model_path.with_suffix(".pt.tmp")
    torch.save(global_model.state_dict(), temporary_model)
    temporary_model.replace(final_model_path)
    final_metrics = dict(round_metrics[-1]) if round_metrics else {}
    total_baseline_energy = sum(row["baseline_round_energy_j"] for row in round_metrics)
    total_proposed_energy = sum(row["round_energy_j"] for row in round_metrics)
    summary = {
        "status": "completed",
        "completed_rounds": completed_round,
        "rounds_requested": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "device": device,
        "seed": seed,
        "budget_fraction": budget_fraction,
        "selection_policy": selection_policy,
        "model_size_mb": model_size_mb,
        "final_metrics": final_metrics,
        "baseline_energy_j": total_baseline_energy,
        "proposed_energy_j": total_proposed_energy,
        "energy_saving_percent": energy_saving_percent(
            total_baseline_energy, total_proposed_energy
        ),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "elapsed_seconds": elapsed_seconds(),
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "global_model": str(final_model_path),
            "client_history": str(history_path),
            "round_metrics": str(metrics_path),
            "progress": str(progress_path),
            "log": str(persistent_log),
        },
    }
    _write_json(summary_path, summary)
    write_progress("completed")
    log(f"Federated training completed: final_f1={final_metrics.get('f1', 0.0):.6f}.")
    return {"history": history, **summary}


def make_incremental_stages(y, n_stages: int = 3) -> list[list[int]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Install project dependencies first: python -m pip install -r requirements.txt") from exc
    labels = np.asarray(y)
    ordered = np.argsort(labels, kind="stable")
    return [part.tolist() for part in np.array_split(ordered, n_stages)]
