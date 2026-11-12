# Running EcoTrust-FLIDS in VS Code or Colab

## Local VS Code Kernel

Use this mode for notebooks `00`, `01`, and `02` when reading datasets from:

`/Volumes/LocalDrive1/Datasets/Security`

Select a local Python kernel on this Mac. A remote or Colab kernel cannot read `/Volumes/LocalDrive1`.

## Colab or Remote Kernel

Remote kernels cannot access the local Mac path. First sync the implementation code to Google Drive:

```bash
cd "/Volumes/LocalDrive1/onlyscholar-projects/EcoTrust-FLIDS[DeepakSir]/IMPLEMENTATION"
bash scripts/sync_code_to_drive.sh
```

Then in Colab, mount Drive and set:

```python
from google.colab import drive
drive.mount('/content/drive')
%env ECOTRUST_PROJECT_ROOT=/content/drive/MyDrive/projects/implementation/EcoTrust-FLIDS[DeepakSir]/IMPLEMENTATION
```

For remote execution, dataset paths also need to point to Drive-mounted datasets, not `/Volumes/LocalDrive1`. Keep local execution until `01_dataset_audit.ipynb` and `02_preprocessing_and_client_splits.ipynb` pass, unless the datasets are mounted in the remote runtime.

## Parallel Publication Execution

Parallel publication runs use only the dedicated notebooks under
`notebooks/parallel/`. Stage 10 has two shards and Stage 12 has three shards.
Assign every shard to a different EcoTrust-specific Colab runtime. Do not run
two copies of the same shard, do not mix shard counts, and do not use a runtime
belonging to another project.

Each shard writes a unique manifest in `logs/` and unique per-run output paths.
The last completed shard automatically validates the full run-key union and
writes the canonical stage manifest. Stage 15 will not start unless the
canonical manifests for stages 09 through 14 are complete.
