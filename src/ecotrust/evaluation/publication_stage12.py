from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
import traceback
from typing import Iterable

from ecotrust.config import load_config, run_mode_settings
from ecotrust.data.clients import ClientSplit
from ecotrust.data.preprocess import load_preprocessed
from ecotrust.evaluation.publication_stage09 import _atomic_json, _peak_rss_mb, _predict_metrics
from ecotrust.evaluation.publication_sharding import (
    merge_completed_shards,
    run_key_digest,
    select_shard,
    shard_manifest_path,
    validate_shard,
)
from ecotrust.federated.client import train_local_model
from ecotrust.federated.server import (
    coordinate_median_state_dicts,
    trimmed_mean_state_dicts,
    weighted_average_state_dicts,
)
from ecotrust.io import read_json, write_json
from ecotrust.models.embedding import load_embedding_model


STAGE = 12
POISONING_PROTOCOL_VERSION = "parameter_updates_only_v2"
EXECUTION_AMENDMENT_VERSION = "cpu_to_gpu_resume_v1"
LOCAL_BATCH_SIZE = 256
LEGACY_V2_CHECKPOINT_DEVICE = "cpu"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_offset(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _extend_device_history(previous: dict | None, device: str) -> list[str]:
    """Preserve execution-backend provenance across durable resumes."""
    history = [] if previous is None else list(previous.get("device_history") or [])
    if previous is not None and not history:
        history.append(str(previous.get("device") or LEGACY_V2_CHECKPOINT_DEVICE))
    if not history or history[-1] != device:
        history.append(device)
    return history


def build_stage12_matrix(stage_cfg: dict) -> list[dict]:
    return [
        {
            "attack": attack,
            "malicious_fraction": float(fraction),
            "aggregator": aggregator,
            "seed": int(seed),
        }
        for attack in stage_cfg["attacks"]
        for fraction in stage_cfg["malicious_fractions"]
        for aggregator in stage_cfg["aggregators"]
        for seed in stage_cfg["matrix_seeds"]
    ]


def _load_checkpoint(torch, path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _poison_state(
    torch,
    state: dict,
    global_state: dict,
    attack: str,
    seed: int,
    parameter_keys: set[str],
) -> dict:
    poisoned = deepcopy(state)
    torch.manual_seed(seed)
    for key, value in poisoned.items():
        reference = global_state[key]
        # Byzantine model-update attacks act on trainable parameters only.
        # BatchNorm running statistics and other floating buffers are not
        # optimization variables; scaling/flipping them can create negative
        # running variances and NaN predictions unrelated to the registered
        # parameter-update attack.
        if key not in parameter_keys:
            poisoned[key] = value
            continue
        if not hasattr(value, "is_floating_point") or not value.is_floating_point():
            poisoned[key] = reference
            continue
        delta = value - reference
        if attack == "sign_flip":
            poisoned[key] = reference - delta
        elif attack == "model_scaling":
            poisoned[key] = reference + 10.0 * delta
        elif attack == "gaussian_update":
            scale = max(float(delta.detach().std().cpu()), 1e-6)
            poisoned[key] = reference + torch.randn_like(delta) * scale
        elif attack == "free_rider":
            poisoned[key] = reference.clone()
        elif attack == "label_flip":
            poisoned[key] = value
        else:
            raise ValueError(f"Unknown attack: {attack}")
    return poisoned


def _validate_aggregated_state(torch, state: dict) -> None:
    for key, value in state.items():
        if not hasattr(value, "is_floating_point") or not value.is_floating_point():
            continue
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Aggregation produced a non-finite tensor for {key!r}")
        if key.endswith("running_var") and bool((value < 0).any()):
            raise FloatingPointError(f"Aggregation produced a negative BatchNorm variance for {key!r}")


def _aggregate(aggregator: str, states: list[tuple[dict, float]], trust_weights: list[float]):
    if aggregator == "fedavg":
        return weighted_average_state_dicts(states)
    if aggregator == "trust_weighted":
        return weighted_average_state_dicts(
            [(state, weight * trust) for (state, weight), trust in zip(states, trust_weights)]
        )
    unweighted = [state for state, _ in states]
    if aggregator == "coordinate_median":
        return coordinate_median_state_dicts(unweighted)
    if aggregator == "trimmed_mean":
        return trimmed_mean_state_dicts(unweighted, trim_fraction=0.2)
    raise ValueError(f"Unknown aggregator: {aggregator}")


def run_single_robustness(
    *,
    config_path: str | Path,
    dataset: str,
    attack: str,
    malicious_fraction: float,
    aggregator: str,
    seed: int,
    run_mode: str = "full",
    device: str = "cpu",
    resume: bool = True,
) -> dict:
    import numpy as np
    import pandas as pd
    import torch

    cfg = load_config(config_path)
    settings = run_mode_settings(cfg, run_mode)
    rounds = int(settings["rounds"])
    local_epochs = int(settings["local_epochs"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Stage 12 requested CUDA, but torch.cuda.is_available() is false")
    # Keep the registered CPU batch size during the authorized GPU continuation.
    # Changing both device and batch size at the resume boundary would alter two
    # execution variables and make the partial-run continuation harder to audit.
    batch_size = LOCAL_BATCH_SIZE
    prep = cfg.results_root / dataset / "preprocessed" / run_mode
    embedding_checkpoint = cfg.results_root / dataset / "embedding" / run_mode / "ecotrust_encoder.pt"
    fraction_id = str(malicious_fraction).replace(".", "p")
    out = (
        cfg.results_root
        / "publication"
        / "stage12"
        / dataset
        / attack
        / f"fraction_{fraction_id}"
        / aggregator
        / f"seed_{seed}"
        / f"protocol_{POISONING_PROTOCOL_VERSION}"
    )
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.pt"
    progress_path = out / "progress.json"
    rounds_path = out / "round_metrics.csv"
    summary_path = out / "summary.json"
    model_path = out / "global_model.pt"
    if resume and summary_path.is_file():
        prior = read_json(summary_path)
        if (
            prior.get("status") == "completed"
            and int(prior.get("completed_rounds", 0)) == rounds
            and prior.get("poisoning_protocol_version") == POISONING_PROTOCOL_VERSION
        ):
            return {**prior, "skipped": True}

    X_train, X_test, y_train, y_test = load_preprocessed(prep)
    splits = [
        ClientSplit(**item)
        for item in read_json(prep / "client_splits.json")
    ]
    rng = np.random.default_rng(seed + _stable_offset(dataset + attack + aggregator + fraction_id))
    malicious_count = max(1, int(round(len(splits) * malicious_fraction)))
    malicious_clients = set(
        rng.choice([split.client_id for split in splits], malicious_count, replace=False).tolist()
    )
    model, _ = load_embedding_model(embedding_checkpoint, device)
    global_model = deepcopy(model).to(device)
    parameter_keys = set(dict(global_model.named_parameters()))
    model_bytes = int(sum(parameter.numel() * parameter.element_size() for parameter in global_model.parameters()))
    completed_round = 0
    round_rows: list[dict] = []
    elapsed_before = 0.0
    started_at = _utc_now()
    device_history = [device]
    if resume and checkpoint_path.is_file():
        checkpoint = _load_checkpoint(torch, checkpoint_path, device)
        protocol = (attack, malicious_fraction, aggregator, seed)
        saved = (
            checkpoint.get("attack"),
            float(checkpoint.get("malicious_fraction", -1)),
            checkpoint.get("aggregator"),
            int(checkpoint.get("seed", -1)),
        )
        if saved != protocol:
            raise RuntimeError(f"Checkpoint protocol mismatch at {checkpoint_path}")
        if checkpoint.get("poisoning_protocol_version") != POISONING_PROTOCOL_VERSION:
            raise RuntimeError(f"Checkpoint poisoning protocol mismatch at {checkpoint_path}")
        saved_batch_size = int(checkpoint.get("batch_size", LOCAL_BATCH_SIZE))
        if saved_batch_size != batch_size:
            raise RuntimeError(
                f"Checkpoint batch-size mismatch at {checkpoint_path}: "
                f"saved={saved_batch_size}, requested={batch_size}"
            )
        global_model.load_state_dict(checkpoint["model_state"])
        completed_round = int(checkpoint["completed_round"])
        round_rows = list(checkpoint.get("round_rows", []))
        device_history = _extend_device_history(checkpoint, device)
        previous_device = device_history[0]
        for row in round_rows:
            row.setdefault("execution_device", previous_device)
        malicious_clients = set(checkpoint.get("malicious_clients", malicious_clients))
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        started_at = str(checkpoint.get("started_at", started_at))

    invocation_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        for round_id in range(completed_round + 1, rounds + 1):
            global_state = deepcopy(global_model.state_dict())
            states: list[tuple[dict, float]] = []
            trust_weights: list[float] = []
            for client_number, split in enumerate(splits):
                local_model = deepcopy(global_model)
                client_y = np.asarray(y_train)[split.indices]
                malicious = split.client_id in malicious_clients
                training_y = 1 - client_y if malicious and attack == "label_flip" else client_y
                state, _ = train_local_model(
                    local_model,
                    X_train.iloc[split.indices],
                    training_y,
                    epochs=local_epochs,
                    batch_size=batch_size,
                    device=device,
                    seed=seed + round_id * 1000 + client_number,
                )
                if malicious:
                    state = _poison_state(
                        torch,
                        state,
                        global_state,
                        attack,
                        seed + round_id * 1000 + client_number + 500000,
                        parameter_keys,
                    )
                states.append((state, float(len(split.indices))))
                trust_weights.append(max(float(split.trust_score), 0.01))
            aggregated_state = _aggregate(aggregator, states, trust_weights)
            _validate_aggregated_state(torch, aggregated_state)
            global_model.load_state_dict(aggregated_state)
            metrics = _predict_metrics(global_model, X_test, y_test, device=device, batch_size=max(batch_size, 2048))
            row = {
                "round": round_id,
                "dataset": dataset,
                "attack": attack,
                "malicious_fraction": malicious_fraction,
                "malicious_clients": malicious_count,
                "aggregator": aggregator,
                "seed": seed,
                "execution_device": device,
                "communication_bytes": int(len(splits) * model_bytes * 2),
                **metrics,
            }
            round_rows.append(row)
            completed_round = round_id
            pd.DataFrame(round_rows).to_csv(rounds_path, index=False)
            checkpoint = {
                "attack": attack,
                "malicious_fraction": malicious_fraction,
                "aggregator": aggregator,
                "seed": seed,
                "poisoning_protocol_version": POISONING_PROTOCOL_VERSION,
                "execution_amendment_version": EXECUTION_AMENDMENT_VERSION,
                "device": device,
                "device_history": device_history,
                "batch_size": batch_size,
                "started_at": started_at,
                "completed_round": completed_round,
                "rounds_requested": rounds,
                "model_state": global_model.state_dict(),
                "round_rows": round_rows,
                "malicious_clients": sorted(malicious_clients),
                "elapsed_seconds": elapsed_before + time.perf_counter() - invocation_started,
            }
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)
            _atomic_json(
                progress_path,
                {
                    "status": "running",
                    "attack": attack,
                    "malicious_fraction": malicious_fraction,
                    "aggregator": aggregator,
                    "seed": seed,
                    "device": device,
                    "device_history": device_history,
                    "batch_size": batch_size,
                    "completed_round": completed_round,
                    "rounds_requested": rounds,
                    "updated_at": _utc_now(),
                },
            )
            print(
                f"{dataset} {attack} fraction={malicious_fraction:g} {aggregator} "
                f"round={round_id}/{rounds} macro_f1={metrics['macro_f1']:.6f}",
                flush=True,
            )
    except BaseException as exc:
        _atomic_json(
            progress_path,
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "attack": attack,
                "malicious_fraction": malicious_fraction,
                "aggregator": aggregator,
                "seed": seed,
                "device": device,
                "device_history": device_history,
                "batch_size": batch_size,
                "completed_round": completed_round,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "updated_at": _utc_now(),
            },
        )
        raise

    torch.save(global_model.state_dict(), model_path)
    summary = {
        "status": "completed",
        "stage": STAGE,
        "dataset": dataset,
        "attack": attack,
        "malicious_fraction": malicious_fraction,
        "aggregator": aggregator,
        "seed": seed,
        "poisoning_protocol_version": POISONING_PROTOCOL_VERSION,
        "execution_amendment_version": EXECUTION_AMENDMENT_VERSION,
        "device": device,
        "device_history": device_history,
        "batch_size": batch_size,
        "malicious_clients": sorted(malicious_clients),
        "completed_rounds": completed_round,
        "rounds_requested": rounds,
        "final_metrics": round_rows[-1],
        "communication_bytes": int(sum(row["communication_bytes"] for row in round_rows)),
        "wall_time_seconds": float(elapsed_before + time.perf_counter() - invocation_started),
        "process_cpu_seconds_this_invocation": float(time.process_time() - cpu_started),
        "peak_rss_mb": _peak_rss_mb(),
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


def run_stage12(
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
    stage_cfg = registry["stages"]["12"]
    selected_datasets = list(datasets or registry["data"]["datasets"])
    matrix = build_stage12_matrix(stage_cfg)
    run_specs = []
    for dataset in selected_datasets:
        for item in matrix:
            fraction_id = str(item["malicious_fraction"]).replace(".", "p")
            key = f"{dataset}/{item['attack']}/fraction_{fraction_id}/{item['aggregator']}/seed_{item['seed']}"
            run_specs.append((key, dataset, item))
    expected_keys = [spec[0] for spec in run_specs]
    assigned_specs = select_shard(run_specs, shard_index, shard_count)
    stem = "12_malicious_client_robustness"
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
        "scope": stage_cfg["scope"],
        "started_at": _utc_now(),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "assigned_run_count": len(assigned_specs),
        "expected_run_count": len(run_specs),
        "expected_run_keys_sha256": run_key_digest(expected_keys),
        "poisoning_protocol_version": POISONING_PROTOCOL_VERSION,
        "execution_amendment_version": EXECUTION_AMENDMENT_VERSION,
        "device_history": [device],
        "batch_size": LOCAL_BATCH_SIZE,
        "runs": {},
    }
    if resume and manifest_path.is_file():
        previous = read_json(manifest_path)
        if previous.get("poisoning_protocol_version") == POISONING_PROTOCOL_VERSION and previous.get("status") in {
            "running",
            "failed",
            "interrupted",
        }:
            manifest["started_at"] = previous.get("started_at", manifest["started_at"])
            manifest["runs"] = previous.get("runs", {})
            manifest["device_history"] = _extend_device_history(previous, device)
        elif previous.get("poisoning_protocol_version") != POISONING_PROTOCOL_VERSION:
            legacy_path = manifest_path.with_name(
                f"{manifest_path.stem}_protocol_v1_buffer_poisoning{manifest_path.suffix}"
            )
            if not legacy_path.is_file():
                write_json(legacy_path, previous)
    write_json(manifest_path, manifest)
    try:
        for key, dataset, item in assigned_specs:
            manifest["runs"][key] = {"status": "running", "updated_at": _utc_now()}
            write_json(manifest_path, manifest)
            result = run_single_robustness(
                config_path=config_path,
                dataset=dataset,
                run_mode=run_mode,
                device=device,
                resume=resume,
                **item,
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
                "scope": stage_cfg["scope"],
                "started_at": manifest["started_at"],
                "poisoning_protocol_version": POISONING_PROTOCOL_VERSION,
                "execution_amendment_version": EXECUTION_AMENDMENT_VERSION,
                "device_history": manifest["device_history"],
                "batch_size": LOCAL_BATCH_SIZE,
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
