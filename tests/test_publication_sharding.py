import json
from pathlib import Path

import pytest

from ecotrust.evaluation.publication_sharding import (
    merge_completed_shards,
    run_key_digest,
    select_shard,
    shard_manifest_path,
    validate_shard,
)


def test_select_shard_is_deterministic_disjoint_and_complete():
    items = list(range(17))
    shards = [select_shard(items, index, 4) for index in range(4)]
    assert [item for shard in shards for item in shard] != items
    assert sorted(item for shard in shards for item in shard) == items
    assert sum(len(shard) for shard in shards) == len(items)
    assert all(set(left).isdisjoint(right) for index, left in enumerate(shards) for right in shards[index + 1 :])


def test_validate_shard_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_shard(0, 0)
    with pytest.raises(ValueError):
        validate_shard(2, 2)
    with pytest.raises(ValueError):
        validate_shard(-1, 2)


def test_merge_completed_shards_requires_exact_coverage(tmp_path):
    keys = [f"run-{index}" for index in range(6)]
    digest = run_key_digest(keys)
    for shard_index in range(2):
        path = shard_manifest_path(tmp_path, "stage", "full", shard_index, 2)
        assigned = select_shard(keys, shard_index, 2)
        path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "shard_index": shard_index,
                    "shard_count": 2,
                    "expected_run_keys_sha256": digest,
                    "runs": {key: {"status": "completed", "summary": {"key": key}} for key in assigned},
                }
            ),
            encoding="utf-8",
        )
    merged = merge_completed_shards(
        logs_root=tmp_path,
        stem="stage",
        run_mode="full",
        shard_count=2,
        expected_keys=keys,
        metadata={"stage": 99, "device": "cpu"},
    )
    assert merged is not None
    assert merged["status"] == "completed"
    assert merged["expected_run_count"] == len(keys)
    assert set(merged["runs"]) == set(keys)
    assert (tmp_path / "stage_full_manifest.json").is_file()


def test_merge_waits_for_every_completed_shard(tmp_path):
    keys = ["a", "b"]
    first = shard_manifest_path(tmp_path, "stage", "full", 0, 2)
    first.write_text(
        json.dumps(
            {
                "status": "completed",
                "shard_index": 0,
                "shard_count": 2,
                "expected_run_keys_sha256": run_key_digest(keys),
                "runs": {"a": {"status": "completed"}},
            }
        ),
        encoding="utf-8",
    )
    assert (
        merge_completed_shards(
            logs_root=tmp_path,
            stem="stage",
            run_mode="full",
            shard_count=2,
            expected_keys=keys,
            metadata={"stage": 99},
        )
        is None
    )


def test_parallel_notebooks_match_registered_shard_plan():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "configs" / "publication_experiments.json").read_text())
    for stage in (10, 12):
        shard_count = int(registry["stages"][str(stage)]["execution"]["shard_count"])
        notebooks = sorted((root / "notebooks" / "parallel").glob(f"{stage}_*_shard_*_of_*.ipynb"))
        assert len(notebooks) == shard_count
        for shard_index, path in enumerate(notebooks):
            notebook = json.loads(path.read_text())
            source = "".join(notebook["cells"][1]["source"])
            assert f"SHARD_INDEX = {shard_index}" in source
            assert f"SHARD_COUNT = {shard_count}" in source
