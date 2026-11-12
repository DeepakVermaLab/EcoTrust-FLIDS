#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="gdrive:projects/implementation/EcoTrust-FLIDS[DeepakSir]/IMPLEMENTATION"
STAGE09_MANIFEST="$REMOTE_ROOT/logs/09_recent_baselines_and_multiseed_full_manifest.json"

command -v rclone >/dev/null || { echo "rclone is required" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }
command -v shasum >/dev/null || { echo "shasum is required" >&2; exit 1; }

stage09_status="$(rclone --tpslimit 1 cat "$STAGE09_MANIFEST" | jq -r '.status')"
if [[ "$stage09_status" != "completed" ]]; then
  echo "Refusing to sync while stage 09 status is '$stage09_status'." >&2
  exit 2
fi

files=(
  "configs/publication_experiments.json"
  "docs/PUBLICATION_EXPERIMENT_REGISTRY.md"
  "docs/RUNNING_IN_VSCODE_OR_COLAB.md"
  "src/ecotrust/evaluation/__init__.py"
  "src/ecotrust/evaluation/publication_sharding.py"
  "src/ecotrust/evaluation/publication_stage10.py"
  "src/ecotrust/evaluation/publication_stage11.py"
  "src/ecotrust/evaluation/publication_stage12.py"
  "src/ecotrust/evaluation/publication_stage13.py"
  "src/ecotrust/evaluation/publication_stage14.py"
  "src/ecotrust/evaluation/publication_stage15.py"
  "notebooks/10_non_iid_dropout_scalability.ipynb"
  "notebooks/11_temporal_drift_and_forgetting.ipynb"
  "notebooks/12_malicious_client_robustness.ipynb"
  "notebooks/13_cross_dataset_controller_generalization.ipynb"
  "notebooks/14_resource_energy_policy_comparison.ipynb"
  "notebooks/15_final_statistics_and_reproducibility.ipynb"
  "notebooks/parallel/10_non_iid_dropout_scalability_shard_01_of_02.ipynb"
  "notebooks/parallel/10_non_iid_dropout_scalability_shard_02_of_02.ipynb"
  "notebooks/parallel/12_malicious_client_robustness_shard_01_of_03.ipynb"
  "notebooks/parallel/12_malicious_client_robustness_shard_02_of_03.ipynb"
  "notebooks/parallel/12_malicious_client_robustness_shard_03_of_03.ipynb"
  "scripts/build_parallel_notebooks.py"
  "tests/test_publication_stage10.py"
  "tests/test_publication_stage11.py"
  "tests/test_publication_stage12.py"
  "tests/test_publication_stage13.py"
  "tests/test_publication_stage14.py"
  "tests/test_publication_stage15.py"
  "tests/test_publication_sharding.py"
)

for relative in "${files[@]}"; do
  local_path="$PROJECT_ROOT/$relative"
  [[ -f "$local_path" ]] || { echo "Missing local file: $local_path" >&2; exit 3; }
done

if [[ "${ECOTRUST_VERIFY_ONLY:-0}" != "1" ]]; then
  for relative in "${files[@]}"; do
    echo "Uploading $relative"
    rclone --tpslimit 1 --retries 5 --low-level-retries 10 copyto \
      "$PROJECT_ROOT/$relative" "$REMOTE_ROOT/$relative"
  done
fi

verification_file="$PROJECT_ROOT/logs/publication_stages_10_15_sync_sha256.tsv"
: > "$verification_file"
verification_dir="$(mktemp -d)"
verification_list="$verification_dir/files.txt"
trap 'rm -rf "$verification_dir"' EXIT
printf '%s\n' "${files[@]}" > "$verification_list"
rclone --tpslimit 4 --retries 5 --low-level-retries 10 copy \
  "$REMOTE_ROOT" "$verification_dir/remote" --files-from "$verification_list"
for relative in "${files[@]}"; do
  local_hash="$(shasum -a 256 "$PROJECT_ROOT/$relative" | awk '{print $1}')"
  remote_hash="$(shasum -a 256 "$verification_dir/remote/$relative" | awk '{print $1}')"
  status="mismatch"
  [[ -n "$remote_hash" && "$local_hash" == "$remote_hash" ]] && status="verified"
  printf '%s\t%s\t%s\t%s\n' "$status" "$local_hash" "$remote_hash" "$relative" | tee -a "$verification_file"
  [[ "$status" == "verified" ]] || exit 4
done

echo "Verified ${#files[@]} publication-stage files."
echo "Verification ledger: $verification_file"
