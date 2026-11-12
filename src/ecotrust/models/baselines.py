from __future__ import annotations

from pathlib import Path


def _deps():
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    except ImportError as exc:
        raise ImportError("Install project dependencies first: python -m pip install -r requirements.txt") from exc
    return joblib, RandomForestClassifier, LogisticRegression, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def train_baselines(X_train, y_train, X_test, y_test, output_dir: str | Path, seed: int = 42) -> dict:
    (
        joblib,
        RandomForestClassifier,
        LogisticRegression,
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    ) = _deps()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    models = {
        "logistic_regression": LogisticRegression(max_iter=500, n_jobs=-1, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=seed,
        ),
    }
    results: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else pred
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            metrics["roc_auc"] = None
        results[name] = metrics
        joblib.dump(model, out / f"{name}.joblib")
    return results
