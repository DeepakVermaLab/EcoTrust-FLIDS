from __future__ import annotations

from copy import deepcopy


def weighted_average_state_dicts(client_states: list[tuple[dict, float]]) -> dict:
    if not client_states:
        raise ValueError("At least one client state is required")
    total_weight = sum(float(w) for _, w in client_states)
    if total_weight <= 0:
        total_weight = float(len(client_states))
        client_states = [(s, 1.0) for s, _ in client_states]
    avg = deepcopy(client_states[0][0])
    for key in avg:
        if not hasattr(avg[key], "is_floating_point") or not avg[key].is_floating_point():
            avg[key] = client_states[0][0][key]
            continue
        avg[key] = avg[key] * (float(client_states[0][1]) / total_weight)
        for state, weight in client_states[1:]:
            avg[key] += state[key] * (float(weight) / total_weight)
    return avg


def coordinate_median_state_dicts(client_states: list[dict]) -> dict:
    if not client_states:
        raise ValueError("At least one client state is required")
    import torch

    result = deepcopy(client_states[0])
    for key in result:
        first = client_states[0][key]
        if not hasattr(first, "is_floating_point") or not first.is_floating_point():
            result[key] = first
            continue
        result[key] = torch.stack([state[key] for state in client_states], dim=0).median(dim=0).values
    return result


def trimmed_mean_state_dicts(client_states: list[dict], trim_fraction: float = 0.2) -> dict:
    if not client_states:
        raise ValueError("At least one client state is required")
    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    import torch

    count = len(client_states)
    trim = int(count * trim_fraction)
    if count - 2 * trim < 1:
        raise ValueError("trim_fraction removes every client update")
    result = deepcopy(client_states[0])
    for key in result:
        first = client_states[0][key]
        if not hasattr(first, "is_floating_point") or not first.is_floating_point():
            result[key] = first
            continue
        stacked = torch.stack([state[key] for state in client_states], dim=0)
        ordered, _ = torch.sort(stacked, dim=0)
        result[key] = ordered[trim : count - trim].mean(dim=0) if trim else ordered.mean(dim=0)
    return result


def evaluate_torch_model(model, X, y, device: str = "cpu", batch_size: int = 4096) -> dict:
    try:
        import numpy as np
        import torch
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except ImportError as exc:
        raise ImportError("Install project dependencies first: python -m pip install -r requirements.txt") from exc
    X_array = np.asarray(X)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_array), batch_size):
            xb = torch.tensor(X_array[start : start + batch_size], dtype=torch.float32).to(device)
            logits = model(xb)
            logits = logits[0] if isinstance(logits, tuple) else logits
            predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
    pred = np.concatenate(predictions) if predictions else np.empty(0, dtype=int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
