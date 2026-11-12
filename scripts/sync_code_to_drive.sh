#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/Volumes/LocalDrive1/onlyscholar-projects/EcoTrust-FLIDS[DeepakSir]/IMPLEMENTATION"
DRIVE_DEST="gdrive:projects/implementation/EcoTrust-FLIDS[DeepakSir]/IMPLEMENTATION"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone is required for Drive sync. Install/configure rclone first." >&2
  exit 1
fi

rclone copy "${PROJECT_ROOT}" "${DRIVE_DEST}" \
  --exclude ".DS_Store" \
  --exclude "._*" \
  --exclude "__pycache__/**" \
  --exclude ".pytest_cache/**" \
  --exclude ".ipynb_checkpoints/**" \
  --exclude ".venv/**" \
  --exclude "results/**" \
  --exclude "logs/**" \
  --exclude "artifacts/**" \
  --progress

echo "Synced implementation code to ${DRIVE_DEST}"
