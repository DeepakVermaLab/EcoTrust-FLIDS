from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyProfile:
    compute_j_per_sample: float = 0.00045
    tx_j_per_mb: float = 0.08
    rx_j_per_mb: float = 0.04
    idle_j_per_round: float = 0.5


def estimate_round_energy(
    n_samples: int,
    model_size_mb: float,
    selected: bool,
    profile: EnergyProfile | None = None,
) -> dict[str, float]:
    p = profile or EnergyProfile()
    if not selected:
        return {"compute_j": 0.0, "tx_j": 0.0, "rx_j": 0.0, "idle_j": p.idle_j_per_round, "total_j": p.idle_j_per_round}
    compute = n_samples * p.compute_j_per_sample
    tx = model_size_mb * p.tx_j_per_mb
    rx = model_size_mb * p.rx_j_per_mb
    total = compute + tx + rx
    return {"compute_j": compute, "tx_j": tx, "rx_j": rx, "idle_j": 0.0, "total_j": total}


def energy_saving_percent(baseline_j: float, proposed_j: float) -> float:
    if baseline_j <= 0:
        return 0.0
    return 100.0 * (baseline_j - proposed_j) / baseline_j
