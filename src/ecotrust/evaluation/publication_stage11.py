from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback
from typing import Iterable

from ecotrust.agent.controller import AgentDecision, ClientState, EcoTrustAgent
from ecotrust.config import load_config, run_mode_settings
from ecotrust.data.clients import dirichlet_client_splits
from ecotrust.data.loaders import detect_label_columns, load_dataset_frame
from ecotrust.data.preprocess import load_preprocessed, make_binary_label
from ecotrust.evaluation.publication_stage09 import _atomic_json, _peak_rss_mb, _predict_metrics
from ecotrust.federated.client import train_local_model
from ecotrust.federated.server import weighted_average_state_dicts
from ecotrust.io import read_json, write_json
from ecotrust.models.embedding import load_embedding_model


STAGE = 11
METHODS = ("naive_fedavg", "reservoir_replay", "ecotrust_class_balanced_replay")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_checkpoint(torch, path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _detect_stream_order(frame, train_indices, windows: int):
    import numpy as np
    import pandas as pd

    candidates = [
        column
        for column in frame.columns
        if any(token in str(column).lower() for token in ("timestamp", "datetime", "date", "time"))
        or str(column).lower() in {"ts", "starttime", "stime"}
    ]
    best = None
    for column in candidates:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            parsed = pd.to_numeric(series, errors="coerce")
        else:
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
            parsed = parsed.map(lambda value: value.timestamp() if not pd.isna(value) else float("nan"))
        subset = parsed.iloc[train_indices]
        valid_fraction = float(subset.notna().mean())
        unique = int(subset.nunique(dropna=True))
        if valid_fraction >= 0.8 and unique >= windows:
            score = (valid_fraction, unique)
            if best is None or score > best[0]:
                best = (score, str(column), subset.to_numpy(dtype=float))
    if best is not None:
        values = best[2]
        finite = np.isfinite(values)
        fill = float(np.nanmax(values[finite])) if finite.any() else 0.0
        ordered_positions = np.argsort(np.where(finite, values, fill), kind="stable")
        return ordered_positions, {
            "order_type": "chronological",
            "order_column": best[1],
            "valid_fraction": best[0][0],
            "unique_values": best[0][1],
        }
    return np.argsort(train_indices, kind="stable"), {
        "order_type": "source_order",
        "order_column": "original_row_position",
        "valid_fraction": None,
        "unique_values": int(len(train_indices)),
    }


def prepare_stream_dataset(config_path: str | Path, dataset: str, run_mode: str, windows: int):
    import numpy as np
    from sklearn.model_selection import train_test_split

    cfg = load_config(config_path)
    prep = cfg.results_root / dataset / "preprocessed" / run_mode
    manifest = read_json(prep / "preprocess_manifest.json")
    X_train, X_test, y_train, y_test = load_preprocessed(prep)
    frame = load_dataset_frame(dataset, cfg, nrows=manifest.get("row_cap"), max_sources=1)
    dataset_cfg = cfg.datasets[dataset]
    label_column, _ = detect_label_columns(frame, dataset_cfg)
    raw_y = make_binary_label(frame[label_column], dataset_cfg.binary_normal_values)
    all_indices = np.arange(len(frame))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=raw_y,
    )
    if not np.array_equal(np.asarray(raw_y)[train_indices], np.asarray(y_train)):
        raise RuntimeError(f"{dataset}: reconstructed stage-02 training membership does not match saved labels")
    if not np.array_equal(np.asarray(raw_y)[test_indices], np.asarray(y_test)):
        raise RuntimeError(f"{dataset}: reconstructed stage-02 test membership does not match saved labels")
    order, protocol = _detect_stream_order(frame, train_indices, windows)
    ordered_X = X_train.iloc[order].reset_index(drop=True)
    ordered_y = np.asarray(y_train)[order]
    positions = np.arange(len(ordered_y))
    window_positions = [np.asarray(part, dtype=int) for part in np.array_split(positions, windows)]
    protocol.update(
        {
            "dataset": dataset,
            "windows": windows,
            "rows": int(len(ordered_y)),
            "window_sizes": [int(len(part)) for part in window_positions],
            "test_rows": int(len(y_test)),
            "test_usage": "final evaluation only",
        }
    )
    return ordered_X, ordered_y, X_test, np.asarray(y_test), window_positions, protocol


def _window_train_validation(indices, labels, validation_fraction: float, seed: int):
    import numpy as np
    from sklearn.model_selection import train_test_split

    local = np.arange(len(indices))
    local_y = np.asarray(labels)[indices]
    counts = np.bincount(local_y.astype(int))
    stratify = local_y if len(np.unique(local_y)) > 1 and len(counts) > 1 and counts.min() >= 2 else None
    train_local, validation_local = train_test_split(
        local,
        test_size=validation_fraction,
        random_state=seed,
        stratify=stratify,
    )
    return indices[train_local], indices[validation_local]


def _update_reservoir(memory_X, memory_y, current_X, current_y, capacity: int, seen: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    if memory_X is None:
        memory_X = np.empty((0, current_X.shape[1]), dtype=current_X.dtype)
        memory_y = np.empty(0, dtype=np.asarray(current_y).dtype)
    X_mem = np.asarray(memory_X).copy()
    y_mem = np.asarray(memory_y).copy()
    for row, label in zip(np.asarray(current_X), np.asarray(current_y)):
        seen += 1
        if len(y_mem) < capacity:
            X_mem = np.vstack([X_mem, row[None, :]])
            y_mem = np.append(y_mem, label)
        else:
            position = int(rng.integers(0, seen))
            if position < capacity:
                X_mem[position] = row
                y_mem[position] = label
    return X_mem, y_mem, seen


def _update_class_balanced(memory_X, memory_y, current_X, current_y, capacity: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    current_X = np.asarray(current_X)
    current_y = np.asarray(current_y)
    if memory_X is None:
        combined_X, combined_y = current_X, current_y
    else:
        combined_X = np.concatenate([np.asarray(memory_X), current_X], axis=0)
        combined_y = np.concatenate([np.asarray(memory_y), current_y], axis=0)
    chosen = []
    labels = sorted(np.unique(combined_y).tolist())
    per_class = max(1, capacity // max(len(labels), 1))
    for label in labels:
        available = np.flatnonzero(combined_y == label)
        chosen.extend(rng.choice(available, min(per_class, len(available)), replace=False).tolist())
    target = min(capacity, len(combined_y))
    if len(chosen) < target:
        remaining = np.setdiff1d(np.arange(len(combined_y)), np.asarray(chosen, dtype=int))
        extra = rng.choice(remaining, min(target - len(chosen), len(remaining)), replace=False)
        chosen.extend(extra.tolist())
    chosen = np.asarray(chosen[:capacity], dtype=int)
    rng.shuffle(chosen)
    return combined_X[chosen], combined_y[chosen]


def _continual_statistics(eval_rows: list[dict], windows: int, rounds_per_window: int) -> dict:
    matrix: list[list[float | None]] = [[None for _ in range(windows)] for _ in range(windows)]
    for row in eval_rows:
        matrix[int(row["after_window"])][int(row["eval_window"])] = float(row["macro_f1"])
    final = matrix[windows - 1]
    if any(value is None for value in final):
        raise ValueError("Final continual-evaluation row is incomplete")
    immediate = [matrix[index][index] for index in range(windows)]
    backward = [float(final[index]) - float(immediate[index]) for index in range(windows - 1)]
    forgetting = []
    recovery_rounds = []
    for task in range(windows - 1):
        trajectory = [matrix[after][task] for after in range(task, windows)]
        if any(value is None for value in trajectory):
            raise ValueError(f"Continual trajectory for window {task} is incomplete")
        best = max(float(value) for value in trajectory)
        forgetting.append(best - float(final[task]))
        threshold = 0.95 * float(trajectory[0])
        later = trajectory[1:]
        first_drop = next((offset for offset, value in enumerate(later) if float(value) < threshold), None)
        if first_drop is None:
            recovery_rounds.append(0)
        else:
            recovered = next(
                (
                    offset
                    for offset, value in enumerate(later[first_drop + 1 :], start=first_drop + 1)
                    if float(value) >= threshold
                ),
                None,
            )
            recovery_rounds.append(None if recovered is None else int((recovered + 1) * rounds_per_window))
    valid_recovery = [value for value in recovery_rounds if value is not None]
    return {
        "average_accuracy": sum(float(value) for value in final) / len(final),
        "backward_transfer": sum(backward) / len(backward) if backward else 0.0,
        "forgetting": sum(forgetting) / len(forgetting) if forgetting else 0.0,
        "recovery_rounds": sum(valid_recovery) / len(valid_recovery) if valid_recovery else None,
        "final_window_macro_f1": [float(value) for value in final],
    }


def run_single_continual(
    *,
    config_path: str | Path,
    dataset: str,
    method: str,
    seed: int,
    stage_cfg: dict,
    stream_data,
    run_mode: str = "full",
    device: str = "cpu",
    resume: bool = True,
) -> dict:
    import numpy as np
    import pandas as pd
    import torch

    if method not in METHODS:
        raise ValueError(f"Unknown stage-11 method: {method}")
    cfg = load_config(config_path)
    settings = run_mode_settings(cfg, run_mode)
    windows = int(stage_cfg["stream_windows"])
    rounds_per_window = max(1, int(settings["rounds"]) // windows)
    local_epochs = int(settings["local_epochs"])
    batch_size = 512 if device.startswith("cuda") else 256
    replay_capacity = int(stage_cfg["replay_capacity"])
    validation_fraction = float(stage_cfg["validation_fraction_per_window"])
    X_train, y_train, X_test, y_test, window_positions, protocol = stream_data
    split_pairs = [
        _window_train_validation(indices, y_train, validation_fraction, seed + window_id)
        for window_id, indices in enumerate(window_positions)
    ]
    train_windows = [pair[0] for pair in split_pairs]
    validation_windows = [pair[1] for pair in split_pairs]
    out = cfg.results_root / "publication" / "stage11" / dataset / method / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.pt"
    progress_path = out / "progress.json"
    eval_path = out / "window_metrics.csv"
    summary_path = out / "summary.json"
    model_path = out / "global_model.pt"
    if resume and summary_path.is_file():
        prior = read_json(summary_path)
        if prior.get("status") == "completed" and int(prior.get("completed_windows", 0)) == windows:
            return {**prior, "skipped": True}

    embedding_checkpoint = cfg.results_root / dataset / "embedding" / run_mode / "ecotrust_encoder.pt"
    model, _ = load_embedding_model(embedding_checkpoint, device)
    global_model = deepcopy(model).to(device)
    completed_windows = 0
    current_window = 0
    completed_round_in_window = 0
    eval_rows: list[dict] = []
    memory_X = None
    memory_y = None
    reservoir_seen = 0
    resume_telemetry = None
    resume_energy_remaining = None
    elapsed_before = 0.0
    started_at = _utc_now()
    if resume and checkpoint_path.is_file():
        checkpoint = _load_checkpoint(torch, checkpoint_path, device)
        if checkpoint.get("method") != method or int(checkpoint.get("seed", -1)) != seed:
            raise RuntimeError(f"Checkpoint protocol mismatch at {checkpoint_path}")
        global_model.load_state_dict(checkpoint["model_state"])
        completed_windows = int(checkpoint.get("completed_windows", 0))
        current_window = int(checkpoint.get("current_window", completed_windows))
        completed_round_in_window = int(checkpoint.get("completed_round_in_window", 0))
        eval_rows = list(checkpoint.get("eval_rows", []))
        memory_X = checkpoint.get("memory_X")
        memory_y = checkpoint.get("memory_y")
        reservoir_seen = int(checkpoint.get("reservoir_seen", 0))
        resume_telemetry = checkpoint.get("telemetry")
        resume_energy_remaining = checkpoint.get("energy_remaining")
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        started_at = str(checkpoint.get("started_at", started_at))

    invocation_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        for window_id in range(completed_windows, windows):
            train_idx = train_windows[window_id]
            current_X = X_train.iloc[train_idx].to_numpy()
            current_y = np.asarray(y_train)[train_idx]
            if method != "naive_fedavg" and memory_X is not None and len(memory_y):
                adapt_X = np.concatenate([current_X, np.asarray(memory_X)], axis=0)
                adapt_y = np.concatenate([current_y, np.asarray(memory_y)], axis=0)
            else:
                adapt_X, adapt_y = current_X, current_y
            splits = dirichlet_client_splits(
                adapt_y,
                n_clients=int(settings["max_clients"]),
                alpha=0.5,
                seed=seed + window_id * 101,
                min_size=20,
            )
            telemetry = {
                split.client_id: {"recent_loss": 0.5, "recent_f1": 0.5, "drift_score": float(window_id > 0)}
                for split in splits
            }
            energy_remaining = {split.client_id: float(split.energy_budget_j) for split in splits}
            if current_window == window_id and completed_round_in_window > 0:
                if resume_telemetry:
                    telemetry.update(resume_telemetry)
                if resume_energy_remaining:
                    energy_remaining.update(resume_energy_remaining)
            agent = EcoTrustAgent()
            start_round = completed_round_in_window + 1 if current_window == window_id else 1
            for round_id in range(start_round, rounds_per_window + 1):
                if method == "ecotrust_class_balanced_replay":
                    decisions = agent.decide(
                        [
                            ClientState(
                                split.client_id,
                                split.trust_score,
                                energy_remaining[split.client_id],
                                telemetry[split.client_id]["recent_loss"],
                                telemetry[split.client_id]["recent_f1"],
                                telemetry[split.client_id]["drift_score"],
                            )
                            for split in splits
                        ],
                        budget_fraction=0.6,
                    )
                else:
                    decisions = [AgentDecision(split.client_id, True, "all", 1.0, 0) for split in splits]
                selected = {decision.client_id for decision in decisions if decision.selected}
                if not selected:
                    selected = {splits[0].client_id}
                client_states = []
                for client_number, split in enumerate(splits):
                    if split.client_id not in selected:
                        continue
                    local_model = deepcopy(global_model)
                    state, local_metrics = train_local_model(
                        local_model,
                        adapt_X[split.indices],
                        adapt_y[split.indices],
                        epochs=local_epochs,
                        batch_size=batch_size,
                        device=device,
                        seed=seed + window_id * 10000 + round_id * 100 + client_number,
                    )
                    weight = float(len(split.indices))
                    if method == "ecotrust_class_balanced_replay":
                        weight *= max(float(split.trust_score), 0.01)
                    client_states.append((state, weight))
                    telemetry[split.client_id]["recent_loss"] = float(local_metrics["loss"])
                    telemetry[split.client_id]["recent_f1"] = float(local_metrics["f1"])
                global_model.load_state_dict(weighted_average_state_dicts(client_states))
                current_window = window_id
                completed_round_in_window = round_id
                checkpoint = {
                    "method": method,
                    "seed": seed,
                    "started_at": started_at,
                    "model_state": global_model.state_dict(),
                    "completed_windows": completed_windows,
                    "current_window": current_window,
                    "completed_round_in_window": completed_round_in_window,
                    "eval_rows": eval_rows,
                    "memory_X": memory_X,
                    "memory_y": memory_y,
                    "reservoir_seen": reservoir_seen,
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
                        "completed_windows": completed_windows,
                        "current_window": current_window,
                        "completed_round_in_window": completed_round_in_window,
                        "rounds_per_window": rounds_per_window,
                        "updated_at": _utc_now(),
                    },
                )
            for eval_window in range(window_id + 1):
                validation_idx = validation_windows[eval_window]
                metrics = _predict_metrics(
                    global_model,
                    X_train.iloc[validation_idx],
                    np.asarray(y_train)[validation_idx],
                    device=device,
                    batch_size=max(batch_size, 2048),
                )
                eval_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "after_window": window_id,
                        "eval_window": eval_window,
                        **metrics,
                    }
                )
            pd.DataFrame(eval_rows).to_csv(eval_path, index=False)
            if method == "reservoir_replay":
                memory_X, memory_y, reservoir_seen = _update_reservoir(
                    memory_X,
                    memory_y,
                    current_X,
                    current_y,
                    replay_capacity,
                    reservoir_seen,
                    seed + window_id,
                )
            elif method == "ecotrust_class_balanced_replay":
                memory_X, memory_y = _update_class_balanced(
                    memory_X,
                    memory_y,
                    current_X,
                    current_y,
                    replay_capacity,
                    seed + window_id,
                )
            completed_windows = window_id + 1
            completed_round_in_window = 0
            current_window = completed_windows
            checkpoint.update(
                {
                    "completed_windows": completed_windows,
                    "current_window": current_window,
                    "completed_round_in_window": 0,
                    "eval_rows": eval_rows,
                    "memory_X": memory_X,
                    "memory_y": memory_y,
                    "reservoir_seen": reservoir_seen,
                    "telemetry": None,
                    "energy_remaining": None,
                    "model_state": global_model.state_dict(),
                }
            )
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)
            print(f"{dataset} {method} seed={seed} window={completed_windows}/{windows}", flush=True)
    except BaseException as exc:
        _atomic_json(
            progress_path,
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "method": method,
                "seed": seed,
                "completed_windows": completed_windows,
                "current_window": current_window,
                "completed_round_in_window": completed_round_in_window,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "updated_at": _utc_now(),
            },
        )
        raise

    final_test = _predict_metrics(global_model, X_test, y_test, device=device, batch_size=max(batch_size, 2048))
    statistics = _continual_statistics(eval_rows, windows, rounds_per_window)
    torch.save(global_model.state_dict(), model_path)
    summary = {
        "status": "completed",
        "stage": STAGE,
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "device": device,
        "completed_windows": completed_windows,
        "stream_protocol": protocol,
        "continual_metrics": statistics,
        "final_test_metrics": final_test,
        "wall_time_seconds": float(elapsed_before + time.perf_counter() - invocation_started),
        "process_cpu_seconds_this_invocation": float(time.process_time() - cpu_started),
        "peak_rss_mb": _peak_rss_mb(),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "window_metrics": str(eval_path),
            "global_model": str(model_path),
            "progress": str(progress_path),
        },
    }
    _atomic_json(summary_path, summary)
    _atomic_json(progress_path, summary)
    return summary


def run_stage11(
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
    stage_cfg = registry["stages"]["11"]
    selected_datasets = list(datasets or registry["data"]["datasets"])
    methods = list(stage_cfg["methods"])
    seeds = [int(seed) for seed in registry["randomness"][stage_cfg["seeds"]]]
    manifest_path = cfg.logs_root / f"11_temporal_drift_and_forgetting_{run_mode}_manifest.json"
    manifest = {
        "stage": STAGE,
        "status": "running",
        "mode": run_mode,
        "device": device,
        "started_at": _utc_now(),
        "runs": {},
        "stream_protocols": {},
    }
    if resume and manifest_path.is_file():
        previous = read_json(manifest_path)
        if previous.get("status") in {"running", "failed", "interrupted"}:
            manifest["started_at"] = previous.get("started_at", manifest["started_at"])
            manifest["runs"] = previous.get("runs", {})
            manifest["stream_protocols"] = previous.get("stream_protocols", {})
    write_json(manifest_path, manifest)
    try:
        for dataset in selected_datasets:
            stream_data = prepare_stream_dataset(
                config_path,
                dataset,
                run_mode,
                int(stage_cfg["stream_windows"]),
            )
            manifest["stream_protocols"][dataset] = stream_data[-1]
            write_json(manifest_path, manifest)
            for method in methods:
                for seed in seeds:
                    key = f"{dataset}/{method}/seed_{seed}"
                    manifest["runs"][key] = {"status": "running", "updated_at": _utc_now()}
                    write_json(manifest_path, manifest)
                    result = run_single_continual(
                        config_path=config_path,
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        stage_cfg=stage_cfg,
                        stream_data=stream_data,
                        run_mode=run_mode,
                        device=device,
                        resume=resume,
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
