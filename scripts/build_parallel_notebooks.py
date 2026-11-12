#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"
OUTPUT_ROOT = NOTEBOOK_ROOT / "parallel"


PLANS = {
    10: {
        "template": NOTEBOOK_ROOT / "10_non_iid_dropout_scalability.ipynb",
        "name": "10_non_iid_dropout_scalability",
        "shard_count": 2,
    },
    12: {
        "template": NOTEBOOK_ROOT / "12_malicious_client_robustness.ipynb",
        "name": "12_malicious_client_robustness",
        "shard_count": 3,
    },
}


def build_notebook(stage: int, shard_index: int, shard_count: int) -> Path:
    plan = PLANS[stage]
    notebook = json.loads(plan["template"].read_text(encoding="utf-8"))
    source = notebook["cells"][1]["source"]
    source = [
        line.replace(
            "SHARD_INDEX = int(os.environ.get('ECOTRUST_SHARD_INDEX', '0'))",
            f"SHARD_INDEX = {shard_index}",
        ).replace(
            "SHARD_COUNT = int(os.environ.get('ECOTRUST_SHARD_COUNT', '1'))",
            f"SHARD_COUNT = {shard_count}",
        )
        for line in source
    ]
    notebook["cells"][1]["source"] = source
    notebook["cells"][0]["source"][0] = (
        f"# {stage} Parallel Shard {shard_index + 1} of {shard_count}\n"
    )
    notebook["cells"][0]["source"].append(
        "\nThis notebook is one deterministic shard of the unchanged registered protocol. "
        "Do not mix notebooks with a different shard count."
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_ROOT / (
        f"{plan['name']}_shard_{shard_index + 1:02d}_of_{shard_count:02d}.ipynb"
    )
    output.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = []
    for stage, plan in PLANS.items():
        shard_count = int(plan["shard_count"])
        for shard_index in range(shard_count):
            output = build_notebook(stage, shard_index, shard_count)
            json.loads(output.read_text(encoding="utf-8"))
            generated.append(output)
    if args.check:
        for output in generated:
            print(output.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
