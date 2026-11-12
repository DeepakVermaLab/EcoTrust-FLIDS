from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Iterable

from ecotrust.agent.controller import AgentDecision, ClientState, EcoTrustAgent
from ecotrust.config import load_config, run_mode_settings
from ecotrust.data.clients import ClientSplit, dirichlet_client_splits
from ecotrust.data.preprocess import load_preprocessed
from ecotrust.energy.simulator import energy_saving_percent, estimate_round_energy
from ecotrust.evaluation.publication_stage09 import _atomic_json, _peak_rss_mb, _predict_metrics
from ecotrust.evaluation.publication_sharding import (
    merge_completed_shards,
    run_key_digest,
    select_shard,
    shard_manifest_path,
    validate_shard,
)
from ecotrust.federated.client import train_local_model
from ecotrust.federated.server import weighted_average_state_dicts
from ecotrust.io import read_json, write_json
from ecotrust.models.embedding import load_embedding_model


STAGE = 10
METHODS = ("fedavg", "ecotrust_agent")


@dataclass(frozen=True)
class StressScenario:
    scenario_id: str
    dirichlet_alpha: float
    client_count: int
    dropout_rate: float
    straggler_rate: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_offset(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def build_stage10_scenarios(stage_cfg: dict, training_cfg: dict) -> list[StressScenario]:
    primary = StressScenario(
        "primary",
        float(training_cfg["dirichlet_alpha"]),
        int(training_cfg["clients"]),
        0.0,
        0.0,
    )
    candidates = [primary]
    candidates.extend(
        StressScenario(f"alpha_{value:g}", float(value), primary.client_count, 0.0, 0.0)
        for value in stage_cfg["dirichlet_alpha"]
    )
    candidates.extend(
        StressScenario(f"clients_{int(value)}", primary.dirichlet_alpha, int(value), 0.0, 0.0)
        for value in stage_cfg["client_counts"]
    )
    candidates.extend(
        StressScenario(f"dropout_{float(value):g}", primary.dirichlet_alpha, primary.client_count, float(value), 0.0)
        for value in stage_cfg["dropout_rates"]
    )
    candidates.extend(
        StressScenario(f"straggler_{float(value):g}", primary.dirichlet_alpha, primary.client_count, 0.0, float(value))
        for value in stage_cfg["straggler_rates"]
    )
    candidates.append(
        StressScenario(
            "hardest_joint",
            min(map(float, stage_cfg["dirichlet_alpha"])),
            max(map(int, stage_cfg["client_counts"])),
            max(map(float, stage_cfg["dropout_rates"])),
            max(map(float, stage_cfg["straggler_rates"])),
        )
    )
    deduplicated: list[StressScenario] = []
    seen: set[tuple[float, int, float, float]] = set()
    for scenario in candidates:
        key = (
            scenario.dirichlet_alpha,
            scenario.client_count,
            scenario.dropout_rate,
            scenario.straggler_rate,
        )
        if key not in seen:
            deduplicated.append(scenario)
            seen.add(key)
    return deduplicated


def _availability_masks(client_ids: list[str], dropout_rate: float, straggler_rate: float, seed: int):
    import random

    rng = random.Random(seed)
    dropped = {client_id for client_id in client_ids if rng.random() < dropout_rate}
    available = [client_id for client_id in client_ids if client_id not in dropped]
    stragglers = {client_id for client_id in available if rng.random() < straggler_rate}
    return dropped, stragglers


def _load_checkpoint(torch, path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_single_stress(
    *,
    config_path: str | Path,
    dataset: str,
    method: str,
    scenario: StressScenario,
    seed: int,
    run_mode: str = "full",
    device: str = "cpu",
    resume: bool = True,
    budget_fraction: float = 0.6,
) -> dict:
    import numpy as np
    import pandas as pd
    import torch

    if method not in METHODS:
        raise ValueError(f"Unknown stage-10 method: {method}")
    cfg = load_config(config_path)
    settings = run_mode_settings(cfg, run_mode)
    rounds = int(settings["rounds"])
    local_epochs = int(settings["local_epochs"])
    batch_size = 512 if device.startswith("cuda") else 256
    prep = cfg.results_root / dataset / "preprocessed" / run_mode
    embedding_checkpoint = cfg.results_root / dataset / "embedding" / run_mode / "ecotrust_encoder.pt"
    out = cfg.results_root / "publication" / "stage10" / dataset / scenario.scenario_id / method / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.pt"
    progress_path = out / "progress.json"
    rounds_path = out / "round_metrics.csv"
    summary_path = out / "summary.json"
    model_path = out / "global_model.pt"
    if resume and summary_path.is_file():
        prior = read_json(summary_path)
        if prior.get("status") == "completed" and int(prior.get("completed_rounds", 0)) == rounds:
            return {**prior, "skipped": True}

    X_train, X_test, y_train, y_test = load_preprocessed(prep)
    splits = dirichlet_client_splits(
        y_train,
        n_clients=scenario.client_count,
        alpha=scenario.dirichlet_alpha,
        seed=seed + _stable_offset(dataset + scenario.scenario_id),
        min_size=20,
    )
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
    elapsed_before = 0.0
    started_at = _utc_now()
    if resume and checkpoint_path.is_file():
        checkpoint = _load_checkpoint(torch, checkpoint_path, device)
        if checkpoint.get("method") != method or checkpoint.get("scenario") != asdict(scenario) or int(checkpoint.get("seed", -1)) != seed:
            raise RuntimeError(f"Checkpoint protocol mismatch at {checkpoint_path}")
        global_model.load_state_dict(checkpoint["model_state"])
        completed_round = int(checkpoint["completed_round"])
        round_rows = list(checkpoint.get("round_rows", []))
        telemetry.update(checkpoint.get("telemetry", {}))
        energy_remaining.update(checkpoint.get("energy_remaining", {}))
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        started_at = str(checkpoint.get("started_at", started_at))

    invocation_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        for round_id in range(completed_round + 1, rounds + 1):
            if method == "ecotrust_agent":
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
            selected = [decision.client_id for decision in decisions if decision.selected]
            if not selected:
                selected = [max(splits, key=lambda split: (split.trust_score, split.client_id)).client_id]
            dropped, stragglers = _availability_masks(
                selected,
                scenario.dropout_rate,
                scenario.straggler_rate,
                seed + round_id * 1009 + _stable_offset(scenario.scenario_id + method),
            )
            trainable = [client_id for client_id in selected if client_id not in dropped]
            contributors = [client_id for client_id in trainable if client_id not in stragglers]
            if not contributors:
                fallback = trainable[0] if trainable else selected[0]
                dropped.discard(fallback)
                stragglers.discard(fallback)
                trainable = list(dict.fromkeys([*trainable, fallback]))
                contributors = [fallback]
            global_state = deepcopy(global_model.state_dict())
            client_states: list[tuple[dict, float]] = []
            round_energy = 0.0
            baseline_energy = 0.0
            for client_number, split in enumerate(splits):
                baseline_energy += estimate_round_energy(len(split.indices), model_size_mb, True)["total_j"]
                if split.client_id in dropped or split.client_id not in selected:
                    continue
                estimated = estimate_round_energy(len(split.indices), model_size_mb, True)
                round_energy += estimated["total_j"]
                energy_remaining[split.client_id] = max(0.0, energy_remaining[split.client_id] - estimated["total_j"])
                local_model = deepcopy(global_model)
                state, local_metrics = train_local_model(
                    local_model,
                    X_train.iloc[split.indices],
                    np.asarray(y_train)[split.indices],
                    epochs=local_epochs,
                    batch_size=batch_size,
                    device=device,
                    seed=seed + round_id * 1000 + client_number,
                )
                telemetry[split.client_id]["recent_loss"] = float(local_metrics["loss"])
                telemetry[split.client_id]["recent_f1"] = float(local_metrics["f1"])
                if split.client_id in contributors:
                    weight = float(len(split.indices))
                    if method == "ecotrust_agent":
                        weight *= max(float(split.trust_score), 0.01)
                    client_states.append((state, weight))
            global_model.load_state_dict(weighted_average_state_dicts(client_states))
            metrics = _predict_metrics(global_model, X_test, y_test, device=device, batch_size=max(batch_size, 2048))
            row = {
                "round": round_id,
                "dataset": dataset,
                "method": method,
                "seed": seed,
                **asdict(scenario),
                "selected_clients": len(selected),
                "dropped_clients": len(dropped),
                "straggler_clients": len(stragglers),
                "contributing_clients": len(contributors),
                "estimated_round_energy_j": round_energy,
                "estimated_all_client_energy_j": baseline_energy,
                "estimated_energy_saving_percent": energy_saving_percent(baseline_energy, round_energy),
                "communication_bytes": int((len(trainable) + len(contributors)) * model_bytes),
                **metrics,
            }
            round_rows.append(row)
            completed_round = round_id
            pd.DataFrame(round_rows).to_csv(rounds_path, index=False)
            checkpoint = {
                "method": method,
                "scenario": asdict(scenario),
                "seed": seed,
                "started_at": started_at,
                "completed_round": completed_round,
                "rounds_requested": rounds,
                "model_state": global_model.state_dict(),
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
                    "scenario": asdict(scenario),
                    "seed": seed,
                    "completed_round": completed_round,
                    "rounds_requested": rounds,
                    "updated_at": _utc_now(),
                },
            )
            print(
                f"{dataset} {scenario.scenario_id} {method} seed={seed} "
                f"round={round_id}/{rounds} macro_f1={metrics['macro_f1']:.6f}",
                flush=True,
            )
    except BaseException as exc:
        _atomic_json(
            progress_path,
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "method": method,
                "scenario": asdict(scenario),
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
        "scenario": asdict(scenario),
        "seed": seed,
        "device": device,
        "completed_rounds": completed_round,
        "rounds_requested": rounds,
        "final_metrics": round_rows[-1],
        "estimated_energy_j": float(sum(row["estimated_round_energy_j"] for row in round_rows)),
        "estimated_all_client_energy_j": float(sum(row["estimated_all_client_energy_j"] for row in round_rows)),
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
    _atomic_json(progress_path, summary)
    return summary


def run_stage10(
    config_path: str | Path,
    registry_path: str | Path,
    *,
    run_mode: str = "full",
    device: str = "cpu",
    datasets: Iterable[str] | None = None,
    resume: bool = True,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict:
    shard_index, shard_count = validate_shard(shard_index, shard_count)
    cfg = load_config(config_path)
    registry = read_json(Path(registry_path))
    stage_cfg = registry["stages"]["10"]
    selected_datasets = list(datasets or registry["data"]["datasets"])
    methods = list(stage_cfg["methods"])
    seeds = [int(seed) for seed in registry["randomness"][stage_cfg["seeds"]]]
    scenarios = build_stage10_scenarios(stage_cfg, registry["training"])
    run_specs = [
        (
            f"{dataset}/{scenario.scenario_id}/{method}/seed_{seed}",
            dataset,
            scenario,
            method,
            seed,
        )
        for dataset in selected_datasets
        for scenario in scenarios
        for method in methods
        for seed in seeds
    ]
    expected_keys = [spec[0] for spec in run_specs]
    assigned_specs = select_shard(run_specs, shard_index, shard_count)
    stem = "10_non_iid_dropout_scalability"
    manifest_path = shard_manifest_path(
        cfg.logs_root,
        stem,
        run_mode,
        shard_index,
        shard_count,
    )
    manifest = {
        "stage": STAGE,
        "status": "running",
        "mode": run_mode,
        "device": device,
        "started_at": _utc_now(),
        "scenario_count": len(scenarios),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "assigned_run_count": len(assigned_specs),
        "expected_run_count": len(run_specs),
        "expected_run_keys_sha256": run_key_digest(expected_keys),
        "runs": {},
    }
    if resume and manifest_path.is_file():
        previous = read_json(manifest_path)
        if previous.get("status") in {"running", "failed", "interrupted"}:
            manifest["started_at"] = previous.get("started_at", manifest["started_at"])
            manifest["runs"] = previous.get("runs", {})
    write_json(manifest_path, manifest)
    try:
        for key, dataset, scenario, method, seed in assigned_specs:
            manifest["runs"][key] = {"status": "running", "updated_at": _utc_now()}
            write_json(manifest_path, manifest)
            result = run_single_stress(
                config_path=config_path,
                dataset=dataset,
                method=method,
                scenario=scenario,
                seed=seed,
                run_mode=run_mode,
                device=device,
                resume=resume,
                budget_fraction=float(registry["training"]["client_fraction"]),
            )
            manifest["runs"][key] = {
                "status": "completed",
                "skipped": bool(result.get("skipped", False)),
                "summary": result,
            }
            write_json(manifest_path, manifest)
        manifest.update({"status": "completed", "completed_at": _utc_now()})
        write_json(manifest_path, manifest)
        merged = merge_completed_shards(
            logs_root=cfg.logs_root,
            stem=stem,
            run_mode=run_mode,
            shard_count=shard_count,
            expected_keys=expected_keys,
            metadata={
                "stage": STAGE,
                "device": device,
                "scenario_count": len(scenarios),
                "started_at": manifest["started_at"],
            },
        )
        return merged or {"manifest": str(manifest_path), **manifest}
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
