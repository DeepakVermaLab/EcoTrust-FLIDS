from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ClientState:
    client_id: str
    trust_score: float
    energy_budget_j: float
    recent_loss: float
    recent_f1: float
    drift_score: float
    queue_delay_ms: float = 0.0


@dataclass(frozen=True)
class AgentDecision:
    client_id: str
    selected: bool
    reason: str
    priority: float
    sleep_rounds: int


class EcoTrustAgent:
    """Rule-based 24x7 controller whose policy is driven by model/trust telemetry.

    This is intentionally deterministic for paper experiments. It can later be
    replaced with a learned policy after the telemetry schema is validated.
    """

    def __init__(
        self,
        min_trust: float = 0.55,
        min_energy_j: float = 120.0,
        drift_boost: float = 0.35,
        max_sleep_rounds: int = 3,
        trust_weight: float = 0.45,
        energy_weight: float = 0.20,
        quality_weight: float = 0.25,
        latency_weight: float = 0.10,
    ) -> None:
        self.min_trust = min_trust
        self.min_energy_j = min_energy_j
        self.drift_boost = drift_boost
        self.max_sleep_rounds = max_sleep_rounds
        self.trust_weight = trust_weight
        self.energy_weight = energy_weight
        self.quality_weight = quality_weight
        self.latency_weight = latency_weight

    def score(self, state: ClientState) -> float:
        energy_term = min(state.energy_budget_j / 1200.0, 1.0)
        quality_term = max(0.0, 1.0 - state.recent_loss) * 0.4 + state.recent_f1 * 0.6
        drift_term = min(state.drift_score, 1.0) * self.drift_boost
        latency_penalty = min(state.queue_delay_ms / 1000.0, 1.0)
        return (
            state.trust_score * self.trust_weight
            + energy_term * self.energy_weight
            + quality_term * self.quality_weight
            + drift_term
            - latency_penalty * self.latency_weight
        )

    def decide(self, states: Iterable[ClientState], budget_fraction: float = 0.6) -> list[AgentDecision]:
        states = list(states)
        if not states:
            return []
        scored = sorted(((self.score(s), s) for s in states), key=lambda x: x[0], reverse=True)
        keep = max(1, int(round(len(scored) * budget_fraction)))
        selected_ids = {s.client_id for _, s in scored[:keep]}
        decisions: list[AgentDecision] = []
        for priority, state in scored:
            if state.trust_score < self.min_trust:
                decisions.append(AgentDecision(state.client_id, False, "low_trust", priority, self.max_sleep_rounds))
            elif state.energy_budget_j < self.min_energy_j:
                decisions.append(AgentDecision(state.client_id, False, "low_energy", priority, self.max_sleep_rounds))
            elif state.client_id in selected_ids:
                decisions.append(AgentDecision(state.client_id, True, "selected", priority, 0))
            else:
                sleep = 1 if state.drift_score > 0.5 else 2
                decisions.append(AgentDecision(state.client_id, False, "energy_save_sleep", priority, sleep))
        return decisions
