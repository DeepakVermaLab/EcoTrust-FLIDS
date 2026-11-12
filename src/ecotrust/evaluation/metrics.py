from __future__ import annotations


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    except ImportError as exc:
        raise ImportError("Install project dependencies first: python -m pip install -r requirements.txt") from exc
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["roc_auc"] = None
    return metrics


def retention_score(old_f1: float, new_f1: float) -> float:
    if old_f1 <= 0:
        return 0.0
    return float(new_f1 / old_f1)
