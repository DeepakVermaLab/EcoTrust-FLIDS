from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
from typing import Iterable

from ecotrust.agent.controller import AgentDecision, ClientState, EcoTrustAgent
from ecotrust.config import load_config, run_mode_settings
from ecotrust.data.clients import ClientSplit
from ecotrust.data.preprocess import load_preprocessed
from ecotrust.evaluation.publication_stage09 import _peak_rss_mb
from ecotrust.evaluation.publication_stage13 import LEARNED_FEATURES, _load_telemetry
from ecotrust.federated.incremental import run_incremental_federated_training
from ecotrust.io import read_json, write_json
from ecotrust.models.embedding import load_embedding_model


STAGE = 14


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _budget_count(total: int, fraction: float) -> int:
    return max(1, min(total, int(round(total * fraction)))) if total else 0


class _MeasuredPolicy:
    def __init__(self, policy: str, budget_fraction: float, seed: int, total_rounds: int, learned_model=None, start_round: int = 0, overhead_seconds: float = 0.0, overhead_path: Path | None = None, initial_energy_by_client: dict[str, float] | None = None):
        self.policy = policy
        self.budget_fraction = budget_fraction
        self.seed = seed
        self.total_rounds = total_rounds
        self.learned_model = learned_model
        self.round_id = start_round
        self.overhead_seconds = float(overhead_seconds)
        self.overhead_path = overhead_path
        self.initial_energy_by_client = initial_energy_by_client or {}
        self._heuristic = EcoTrustAgent()

    def _record_overhead(self, started: float) -> None:
        self.overhead_seconds += time.perf_counter() - started
        if self.overhead_path is not None:
            write_json(
                self.overhead_path,
                {
                    "policy": self.policy,
                    "completed_decisions": self.round_id,
                    "controller_overhead_seconds": self.overhead_seconds,
                    "updated_at": _utc_now(),
                },
            )

    def decide(self, states, budget_fraction: float | None = None):
        import numpy as np
        import pandas as pd

        started = time.perf_counter()
        states = list(states)
        self.round_id += 1
        fraction = self.budget_fraction if budget_fraction is None else budget_fraction
        count = _budget_count(len(states), fraction)
        if self.policy == "all_clients":
            selected = {state.client_id for state in states}
            priorities = {state.client_id: 1.0 for state in states}
        elif self.policy == "random_budget":
            rng = np.random.default_rng(self.seed + self.round_id)
            chosen = rng.choice(len(states), count, replace=False)
            selected = {states[int(index)].client_id for index in chosen}
            priorities = {state.client_id: 1.0 if state.client_id in selected else 0.0 for state in states}
        elif self.policy == "round_robin":
            ordered = sorted(states, key=lambda state: state.client_id)
            offset = ((self.round_id - 1) * count) % len(ordered)
            selected = {ordered[(offset + index) % len(ordered)].client_id for index in range(count)}
            priorities = {state.client_id: 1.0 if state.client_id in selected else 0.0 for state in states}
        elif self.policy == "static_score":
            ordered = sorted(states, key=lambda state: (state.trust_score, state.client_id), reverse=True)
            selected = {state.client_id for state in ordered[:count]}
            priorities = {state.client_id: state.trust_score for state in states}
        elif self.policy == "telemetry_heuristic":
            decisions = self._heuristic.decide(states, budget_fraction=fraction)
            self._record_overhead(started)
            return decisions
        elif self.policy == "learned_policy":
            if self.learned_model is None:
                raise ValueError("learned_model is required for learned_policy")
            features = pd.DataFrame(
                [
                    {
                        "trust_score": state.trust_score,
                        "drift_score": state.drift_score,
                        "remaining_energy_fraction": min(
                            max(
                                state.energy_budget_j
                                / max(float(self.initial_energy_by_client.get(state.client_id, state.energy_budget_j)), 1e-12),
                                0.0,
                            ),
                            1.0,
                        ),
                        "round_fraction": self.round_id / max(self.total_rounds, 1),
                    }
                    for state in states
                ]
            )
            scores = self.learned_model.predict_proba(features[list(LEARNED_FEATURES)])[:, 1]
            order = np.argsort(-scores, kind="stable")
            selected = {states[int(index)].client_id for index in order[:count]}
            priorities = {state.client_id: float(scores[index]) for index, state in enumerate(states)}
        else:
            raise ValueError(f"Unknown stage-14 policy: {self.policy}")
        decisions = [
            AgentDecision(
                state.client_id,
                state.client_id in selected,
                self.policy,
                float(priorities[state.client_id]),
                0 if state.client_id in selected else 1,
            )
            for state in states
        ]
        self._record_overhead(started)
        return decisions


class _EnergyProbe:
    def __init__(self):
        self.source = None
        self.path = None
        self.nvml = None
        self.nvml_handle = None
        candidates = sorted(Path("/sys/class/powercap").glob("intel-rapl*/energy_uj")) if Path("/sys/class/powercap").exists() else []
        if candidates:
            self.source = "rapl"
            self.path = candidates[0]
        else:
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                self.source = "nvml"
                self.nvml = pynvml
                self.nvml_handle = handle
            except Exception:
                self.source = None

    def read_joules(self):
        if self.source == "rapl" and self.path is not None:
            try:
                return float(self.path.read_text(encoding="utf-8").strip()) / 1_000_000.0
            except (OSError, ValueError):
                return None
        if self.source == "nvml" and self.nvml is not None and self.nvml_handle is not None:
            try:
                return float(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.nvml_handle)) / 1000.0
            except Exception:
                return None
        return None


def _fit_learned_policy(cfg, held_out: str, datasets: list[str], run_mode: str, seed: int):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    frames = [_load_telemetry(cfg, dataset, run_mode)[0] for dataset in datasets if dataset != held_out]
    training = pd.concat(frames, ignore_index=True)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    model.fit(training[list(LEARNED_FEATURES)], training["selected"].astype(int))
    return model


def run_single_policy(
    *,
    config_path: str | Path,
    registry_path: str | Path,
    dataset: str,
    policy: str,
    seed: int,
    run_mode: str,
    device: str,
    resume: bool,
) -> dict:
    import pandas as pd

    cfg = load_config(config_path)
    registry = read_json(Path(registry_path))
    settings = run_mode_settings(cfg, run_mode)
    rounds = int(settings["rounds"])
    local_epochs = int(settings["local_epochs"])
    batch_size = 512 if device.startswith("cuda") else 256
    budget_fraction = 1.0 if policy == "all_clients" else float(registry["training"]["client_fraction"])
    output_dir = cfg.results_root / "publication" / "stage14" / dataset / policy / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if resume and summary_path.is_file():
        prior = read_json(summary_path)
        if prior.get("status") == "completed" and int(prior.get("completed_rounds", 0)) == rounds:
            return {**prior, "skipped": True}

    prep = cfg.results_root / dataset / "preprocessed" / run_mode
    X_train, X_test, y_train, y_test = load_preprocessed(prep)
    splits = [ClientSplit(**item) for item in read_json(prep / "client_splits.json")]
    initial_energy_by_client = {split.client_id: float(split.energy_budget_j) for split in splits}
    embedding_checkpoint = cfg.results_root / dataset / "embedding" / run_mode / "ecotrust_encoder.pt"
    model, _ = load_embedding_model(embedding_checkpoint, device)
    learned_model = None
    if policy == "learned_policy":
        learned_model = _fit_learned_policy(cfg, dataset, list(registry["data"]["datasets"]), run_mode, seed)
    progress_path = output_dir / "federated_progress.json"
    overhead_path = output_dir / "controller_overhead.json"
    start_round = int(read_json(progress_path).get("completed_round", 0)) if resume and progress_path.is_file() else 0
    prior_overhead = read_json(overhead_path).get("controller_overhead_seconds", 0.0) if resume and overhead_path.is_file() else 0.0
    policy_agent = _MeasuredPolicy(
        policy,
        budget_fraction,
        seed,
        rounds,
        learned_model,
        start_round,
        float(prior_overhead),
        overhead_path,
        initial_energy_by_client,
    )
    energy_probe = _EnergyProbe()
    direct_start = energy_probe.read_joules()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
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
        batch_size=batch_size,
        device=device,
        seed=seed,
        resume=resume,
        selection_policy="agent",
        budget_fraction=budget_fraction,
        agent=policy_agent,
    )
    direct_end = energy_probe.read_joules()
    history = pd.read_csv(output_dir / "federated_round_history.csv")
    model_bytes = int(round(float(result["model_size_mb"]) * 1024**2))
    if history["selected"].dtype == bool:
        selected_updates = int(history["selected"].sum())
    else:
        selected_updates = int(history["selected"].astype(str).str.lower().eq("true").sum())
    communication_bytes = int(selected_updates * model_bytes * 2)
    direct_energy = None
    uninterrupted_full_run = start_round == 0 and int(result["completed_rounds"]) == rounds
    if uninterrupted_full_run and direct_start is not None and direct_end is not None and direct_end >= direct_start:
        direct_energy = direct_end - direct_start
    process_cpu_seconds = float(time.process_time() - cpu_started)
    wall_time_seconds_this_invocation = float(time.perf_counter() - wall_started)
    summary = {
        "status": "completed",
        "stage": STAGE,
        "dataset": dataset,
        "policy": policy,
        "seed": seed,
        "device": device,
        "completed_rounds": int(result["completed_rounds"]),
        "rounds_requested": rounds,
        "final_metrics": result["final_metrics"],
        "estimated_energy_j": float(result["proposed_energy_j"]),
        "estimated_all_client_energy_j": float(result["baseline_energy_j"]),
        "estimated_energy_saving_percent": float(result["energy_saving_percent"]),
        "energy_measurement_type": "direct" if direct_energy is not None else "model_estimate",
        "direct_energy_source": energy_probe.source if direct_energy is not None else None,
        "direct_energy_j": direct_energy,
        "energy_measurement_scope": "single_uninterrupted_run" if direct_energy is not None else "model_estimate",
        "wall_time_seconds": float(result["elapsed_seconds"]),
        "wall_time_seconds_this_invocation": wall_time_seconds_this_invocation,
        "process_cpu_seconds": process_cpu_seconds,
        "process_cpu_seconds_this_invocation": process_cpu_seconds,
        "resource_measurement_scope": "single_uninterrupted_run" if uninterrupted_full_run else "cumulative_wall_with_resume_invocation_cpu",
        "peak_rss_mb": _peak_rss_mb(),
        "communication_bytes": communication_bytes,
        "controller_overhead_seconds": float(policy_agent.overhead_seconds),
        "controller_overhead_seconds_this_invocation": float(policy_agent.overhead_seconds - prior_overhead),
        "completed_at": _utc_now(),
        "artifacts": {**result["artifacts"], "controller_overhead": str(overhead_path)},
    }
    write_json(summary_path, summary)
    return summary


def run_stage14(
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
    stage_cfg = registry["stages"]["14"]
    selected_datasets = list(datasets or registry["data"]["datasets"])
    seed = int(stage_cfg["comparison_seed"])
    manifest_path = cfg.logs_root / f"14_resource_energy_policy_comparison_{run_mode}_manifest.json"
    manifest = {
        "stage": STAGE,
        "status": "running",
        "mode": run_mode,
        "device": device,
        "scope": stage_cfg["scope"],
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
            for policy in stage_cfg["policies"]:
                key = f"{dataset}/{policy}/seed_{seed}"
                manifest["runs"][key] = {"status": "running", "updated_at": _utc_now()}
                write_json(manifest_path, manifest)
                result = run_single_policy(
                    config_path=config_path,
                    registry_path=registry_path,
                    dataset=dataset,
                    policy=policy,
                    seed=seed,
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
