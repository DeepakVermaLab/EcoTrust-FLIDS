from __future__ import annotations

from pathlib import Path


def permutation_importance_report(model, X, y, feature_names: list[str], output_dir: str | Path, n_repeats: int = 5) -> dict:
    try:
        import pandas as pd
        from sklearn.inspection import permutation_importance
    except ImportError as exc:
        raise ImportError("Install project dependencies first: python -m pip install -r requirements.txt") from exc
    result = permutation_importance(model, X, y, n_repeats=n_repeats, scoring="f1", n_jobs=-1, random_state=42)
    rows = [
        {"feature": feature_names[i], "importance_mean": float(result.importances_mean[i]), "importance_std": float(result.importances_std[i])}
        for i in result.importances_mean.argsort()[::-1]
    ]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "permutation_importance.csv", index=False)
    return {"top_features": rows[:25]}
