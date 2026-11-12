from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "project.json"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    label_column: str
    attack_type_column: str | None
    binary_normal_values: tuple[Any, ...]
    preferred_files: tuple[str, ...] = ()
    preferred_archives: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]
    project_root: Path
    dataset_root: Path
    results_root: Path
    logs_root: Path
    drive_output_folder_id: str
    drive_output_url: str
    seed: int
    default_device: str
    datasets: dict[str, DatasetConfig]


def load_config(path: str | Path | None = None) -> ProjectConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.expanduser().open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    datasets: dict[str, DatasetConfig] = {}
    for name, item in raw["datasets"].items():
        datasets[name] = DatasetConfig(
            name=name,
            path=Path(item["path"]),
            label_column=item["label_column"],
            attack_type_column=item.get("attack_type_column"),
            binary_normal_values=tuple(item.get("binary_normal_values", [])),
            preferred_files=tuple(item.get("preferred_files", [])),
            preferred_archives=tuple(item.get("preferred_archives", [])),
        )

    return ProjectConfig(
        raw=raw,
        project_root=Path(raw["local_project_root"]),
        dataset_root=Path(raw["local_dataset_root"]),
        results_root=Path(raw["local_results_root"]),
        logs_root=Path(raw["local_logs_root"]),
        drive_output_folder_id=raw["drive_output_folder_id"],
        drive_output_url=raw["drive_output_url"],
        seed=int(raw.get("seed", 42)),
        default_device=raw.get("default_device", "cpu"),
        datasets=datasets,
    )


def ensure_project_dirs(config: ProjectConfig) -> None:
    for path in [config.results_root, config.logs_root]:
        path.mkdir(parents=True, exist_ok=True)
    for dataset in config.datasets:
        (config.results_root / dataset).mkdir(parents=True, exist_ok=True)


def run_mode_settings(config: ProjectConfig, mode: str) -> dict[str, Any]:
    modes = config.raw.get("run_modes", {})
    if mode not in modes:
        raise KeyError(f"Unknown run mode {mode!r}. Available modes: {sorted(modes)}")
    return dict(modes[mode])
