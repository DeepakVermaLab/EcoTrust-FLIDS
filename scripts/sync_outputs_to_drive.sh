#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/Volumes/LocalDrive1/onlyscholar-projects/EcoTrust-FLIDS[DeepakSir]/IMPLEMENTATION"
CONFIG_PATH="${PROJECT_ROOT}/configs/project.json"
FOLDER_ID="$(python3 - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path('/Volumes/LocalDrive1/onlyscholar-projects/EcoTrust-FLIDS[DeepakSir]/IMPLEMENTATION/configs/project.json').read_text())
print(cfg['drive_output_folder_id'])
PY
)"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone is required for Drive sync. Install/configure rclone first." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

rclone copy "results" "gdrive:results" \
  --drive-root-folder-id "${FOLDER_ID}" \
  --create-empty-src-dirs \
  --exclude ".DS_Store" \
  --exclude "._*" \
  --progress

rclone copy "logs" "gdrive:logs" \
  --drive-root-folder-id "${FOLDER_ID}" \
  --create-empty-src-dirs \
  --exclude ".DS_Store" \
  --exclude "._*" \
  --progress

echo "Synced generated outputs to Google Drive folder ${FOLDER_ID}"
