from __future__ import annotations

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_baseline(summary: dict, dataset: str, rounds: int) -> None:
    if summary.get("status") != "completed":
        raise RuntimeError(f"{dataset}: stage-05 baseline is not completed")
    if summary.get("selection_policy") != "all":
        raise RuntimeError(f"{dataset}: stage-05 baseline is not the all-client policy")
    if int(summary.get("completed_rounds", 0)) != rounds:
        raise RuntimeError(
            f"{dataset}: stage-05 baseline has {summary.get('completed_rounds')} rounds; expected {rounds}"
        )


def _comparison(baseline: dict, agentic: dict, budget_fraction: float) -> dict:
    baseline_metrics = baseline["final_metrics"]
    agentic_metrics = agentic["final_metrics"]
    baseline_energy = float(baseline["proposed_energy_j"])
    agentic_energy = float(agentic["proposed_energy_j"])
    energy_saving = 0.0 if baseline_energy <= 0 else 100.0 * (
        baseline_energy - agentic_energy
    ) / baseline_energy
    return {
        "status": "completed",
        "policy": "telemetry_agent",
        "matched_baseline_policy": "all",
        "budget_fraction": budget_fraction,
        "rounds": int(agentic["completed_rounds"]),
        "baseline_final_metrics": baseline_metrics,
        "agentic_final_metrics": agentic_metrics,
        "f1_delta": float(agentic_metrics["f1"]) - float(baseline_metrics["f1"]),
        "accuracy_delta": float(agentic_metrics["accuracy"]) - float(baseline_metrics["accuracy"]),
        "baseline_energy_j": baseline_energy,
        "agentic_energy_j": agentic_energy,
        "energy_saving_percent": energy_saving,
        "baseline_elapsed_seconds": float(baseline.get("elapsed_seconds", 0.0)),
        "agentic_elapsed_seconds": float(agentic.get("elapsed_seconds", 0.0)),
        "completed_at": _utc_now(),
    }


def run_agentic_energy_experiment(
    config_path: str | Path,
    *,
    run_mode: str = "full",
    datasets: Iterable[str] = ("Edge-IIoTset", "TON_IoT", "CICIoT2023"),
    device: str = "cpu",
    budget_fraction: float = 0.6,
    batch_size: int | None = None,
    resume: bool = True,
) -> dict:
    """Run the telemetry-driven controller against the matched stage-05 baseline."""

    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be 'cpu' or a CUDA device")
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")

    cfg = load_config(config_path)
    settings = run_mode_settings(cfg, run_mode)
    rounds = int(settings["rounds"])
    local_epochs = int(settings["local_epochs"])
    effective_batch_size = batch_size or (512 if device.startswith("cuda") else 256)
    run_log = cfg.logs_root / f"06_agentic_energy_controller_{run_mode}.log"
    manifest_path = cfg.logs_root / f"06_agentic_energy_controller_{run_mode}_manifest.json"
    cfg.logs_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": 6,
        "mode": run_mode,
        "device": device,
        "budget_fraction": budget_fraction,
        "started_at": _utc_now(),
        "datasets": {},
    }
    write_json(manifest_path, manifest)

    for dataset in datasets:
        baseline_dir = cfg.results_root / dataset / "federated" / run_mode
        baseline_summary_path = baseline_dir / "federated_summary.json"
        if not baseline_summary_path.exists():
            raise FileNotFoundError(f"{dataset}: missing stage-05 summary: {baseline_summary_path}")
        baseline_summary = read_json(baseline_summary_path)
        _validate_baseline(baseline_summary, dataset, rounds)

        output_dir = cfg.results_root / dataset / "agentic" / run_mode
        comparison_path = output_dir / "baseline_agentic_comparison.json"
        agentic_summary_path = output_dir / "federated_summary.json"
        if resume and comparison_path.exists() and agentic_summary_path.exists():
            prior_agentic = read_json(agentic_summary_path)
            prior_comparison = read_json(comparison_path)
            if (
                prior_agentic.get("status") == "completed"
                and int(prior_agentic.get("completed_rounds", 0)) == rounds
                and float(prior_agentic.get("budget_fraction", -1.0)) == budget_fraction
                and prior_comparison.get("status") == "completed"
            ):
                manifest["datasets"][dataset] = {
                    "status": "completed",
                    "skipped": True,
                    "comparison": prior_comparison,
                }
                write_json(manifest_path, manifest)
                print("Already completed; skipping", dataset, flush=True)
                continue

        prep = cfg.results_root / dataset / "preprocessed" / run_mode
        embedding_checkpoint = (
            cfg.results_root / dataset / "embedding" / run_mode / "ecotrust_encoder.pt"
        )
        X_train, X_test, y_train, y_test = load_preprocessed(prep)
        splits = [
            ClientSplit(**item)
            for item in json.loads((prep / "client_splits.json").read_text(encoding="utf-8"))
        ]
        model, checkpoint = load_embedding_model(embedding_checkpoint, device)
        if int(checkpoint["config"]["input_dim"]) != X_train.shape[1]:
            raise ValueError(f"{dataset}: encoder input dimension does not match preprocessed features")

        manifest["datasets"][dataset] = {
            "status": "running",
            "baseline_summary": str(baseline_summary_path),
            "output_dir": str(output_dir),
        }
        write_json(manifest_path, manifest)
        print("Agentic federated", dataset, flush=True)
        try:
            result = run_incremental_federated_training(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                splits,
                output_dir,
                rounds=rounds,
                local_epochs=local_epochs,
                batch_size=effective_batch_size,
                device=device,
                seed=cfg.seed,
                resume=resume,
                log_path=run_log,
                selection_policy="agent",
                budget_fraction=budget_fraction,
                agent=EcoTrustAgent(),
            )
            comparison = _comparison(baseline_summary, result, budget_fraction)
            write_json(comparison_path, comparison)
            manifest["datasets"][dataset] = {
                "status": "completed",
                "output_dir": str(output_dir),
                "comparison": comparison,
            }
            write_json(manifest_path, manifest)
            print(dataset, comparison, flush=True)
        except BaseException as exc:
            manifest["datasets"][dataset] = {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "output_dir": str(output_dir),
            }
            write_json(manifest_path, manifest)
            raise

    manifest["status"] = "completed"
    manifest["completed_at"] = _utc_now()
    write_json(manifest_path, manifest)
    return {"manifest": str(manifest_path), **manifest}
