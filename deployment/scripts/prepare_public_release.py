#!/usr/bin/env python3
"""Prepare a path-sanitized, GitHub-Pages-ready simulation repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
DIST = DEPLOYMENT_ROOT / "dist"
SCREENSHOTS = DEPLOYMENT_ROOT / "screenshots"
README = DEPLOYMENT_ROOT / "PUBLIC_README.md"

REPLACEMENTS = {
    "/content/drive/MyDrive/projects/implementation/EcoTrust-FLIDS[DeepakSir]/IMPLEMENTATION/": "project://",
    "/Volumes/LocalDrive1/onlyscholar-projects/EcoTrust-FLIDS[DeepakSir]/IMPLEMENTATION/": "project://",
    "/Volumes/LocalDrive1/Datasets/Security/": "dataset://",
}

TEXT_SUFFIXES = {".csv", ".html", ".js", ".json", ".md", ".sha256", ".txt"}


def is_sidecar(path: Path) -> bool:
    return path.name == ".DS_Store" or path.name.startswith("._")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for source, replacement in REPLACEMENTS.items():
        text = text.replace(source, replacement)
    path.write_text(text, encoding="utf-8")


def download_records(downloads: Path) -> list[dict[str, object]]:
    provenance = downloads / "evidence_manifest.json"
    paths = sorted(
        path
        for path in downloads.rglob("*")
        if path.is_file() and path != provenance and not is_sidecar(path)
    )
    records = [
        {
            "file": str(path.relative_to(downloads.parent)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]
    payload = {
        "source": "accepted EcoTrust-FLIDS Stage 09-15 artifacts",
        "sanitization": "internal filesystem paths replaced with portable project:// references",
        "files": records,
    }
    provenance.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    records.append(
        {
            "file": str(provenance.relative_to(downloads.parent)),
            "bytes": provenance.stat().st_size,
            "sha256": sha256(provenance),
        }
    )
    return records


def update_data_bundle(output: Path, records: list[dict[str, object]]) -> None:
    data_path = output / "data.js"
    prefix = "window.ECOTRUST_DATA = "
    text = data_path.read_text(encoding="utf-8").strip()
    if not text.startswith(prefix) or not text.endswith(";"):
        raise RuntimeError(f"Unexpected data bundle format: {data_path}")
    payload = json.loads(text[len(prefix) : -1])
    payload["downloads"] = records
    data_path.write_text(prefix + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")


def verify_release(output: Path) -> None:
    required = ("README.md", "index.html", "styles.css", "app.js", "data.js", ".nojekyll")
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Public release is missing required files: {missing}")
    forbidden = tuple(REPLACEMENTS)
    for path in output.rglob("*"):
        if not path.is_file() or is_sidecar(path) or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        matches = [value for value in forbidden if value in text]
        if matches:
            raise RuntimeError(f"Internal path remained in {path}: {matches}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="Empty or existing release directory")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    ignore = shutil.ignore_patterns("._*", ".DS_Store")
    shutil.copytree(
        DIST,
        output,
        dirs_exist_ok=True,
        ignore=ignore,
        copy_function=shutil.copyfile,
    )
    shutil.copytree(
        SCREENSHOTS,
        output / "screenshots",
        dirs_exist_ok=True,
        ignore=ignore,
        copy_function=shutil.copyfile,
    )
    shutil.copyfile(README, output / "README.md")
    (output / ".gitignore").write_text(".DS_Store\n._*\n", encoding="utf-8")

    for path in output.rglob("*"):
        if path.is_file() and not is_sidecar(path) and path.suffix.lower() in TEXT_SUFFIXES:
            sanitize_text(path)

    records = download_records(output / "downloads")
    update_data_bundle(output, records)
    verify_release(output)

    print("PUBLIC_RELEASE_STATUS: ready")
    print(f"output: {output}")
    print(f"download_files: {len(records)}")


if __name__ == "__main__":
    main()
