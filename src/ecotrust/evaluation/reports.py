from __future__ import annotations

from pathlib import Path

from ecotrust.io import dataframe_to_table, utc_now_iso, write_json, write_text


def write_metrics_report(output_dir: str | Path, title: str, metrics: dict) -> Path:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Install project dependencies first: python -m pip install -r requirements.txt") from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, values in metrics.items():
        row = {"method": method}
        row.update(values)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    write_json(out / "metrics.json", {"created_at_utc": utc_now_iso(), "metrics": metrics})
    md = f"# {title}\n\nCreated: {utc_now_iso()}\n\n{dataframe_to_table(df)}\n"
    return write_text(out / "REPORT.md", md)
