from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence, TypeVar


T = TypeVar("T")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_shard(shard_index: int, shard_count: int) -> tuple[int, int]:
    shard_index = int(shard_index)
    shard_count = int(shard_count)
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count - 1}]")
    return shard_index, shard_count


def select_shard(items: Sequence[T], shard_index: int, shard_count: int) -> list[T]:
    shard_index, shard_count = validate_shard(shard_index, shard_count)
    return [item for position, item in enumerate(items) if position % shard_count == shard_index]


def run_key_digest(keys: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(key) for key in keys)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def shard_manifest_path(
    logs_root: Path,
    stem: str,
    run_mode: str,
    shard_index: int,
    shard_count: int,
) -> Path:
    shard_index, shard_count = validate_shard(shard_index, shard_count)
    if shard_count == 1:
        return logs_root / f"{stem}_{run_mode}_manifest.json"
    return logs_root / (
        f"{stem}_{run_mode}_shard_{shard_index + 1:02d}_of_{shard_count:02d}_manifest.json"
    )


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def merge_completed_shards(
    *,
    logs_root: Path,
    stem: str,
    run_mode: str,
    shard_count: int,
    expected_keys: Sequence[str],
    metadata: dict,
) -> dict | None:
    _, shard_count = validate_shard(0, shard_count)
    expected = set(expected_keys)
    expected_digest = run_key_digest(expected)
    merged_runs: dict[str, dict] = {}
    shard_paths: list[Path] = []
    for shard_index in range(shard_count):
        path = shard_manifest_path(logs_root, stem, run_mode, shard_index, shard_count)
        shard_paths.append(path)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            return None
        if int(payload.get("shard_index", -1)) != shard_index:
            raise RuntimeError(f"Shard index mismatch in {path}")
        if int(payload.get("shard_count", -1)) != shard_count:
            raise RuntimeError(f"Shard count mismatch in {path}")
        if payload.get("expected_run_keys_sha256") != expected_digest:
            raise RuntimeError(f"Registered run-key digest mismatch in {path}")
        for key, result in payload.get("runs", {}).items():
            if key in merged_runs:
                raise RuntimeError(f"Duplicate run key across shards: {key}")
            if result.get("status") != "completed":
                raise RuntimeError(f"Non-completed run in completed shard {path}: {key}")
            merged_runs[key] = result
    actual = set(merged_runs)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"Shard coverage mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    canonical = logs_root / f"{stem}_{run_mode}_manifest.json"
    merged = {
        **metadata,
        "status": "completed",
        "mode": run_mode,
        "sharded": shard_count > 1,
        "shard_count": shard_count,
        "expected_run_count": len(expected),
        "expected_run_keys_sha256": expected_digest,
        "shard_manifests": [str(path) for path in shard_paths],
        "runs": merged_runs,
        "completed_at": utc_now(),
    }
    atomic_json(canonical, merged)
    return {"manifest": str(canonical), **merged}
