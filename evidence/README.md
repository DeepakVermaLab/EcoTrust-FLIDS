# Accepted experiment evidence

This directory contains compact, reviewable evidence for the completed EcoTrust-FLIDS pipeline. It is intentionally separate from generated `results/` and `logs/`, which may contain large checkpoints, preprocessed matrices, and transient execution state.

## Contents

- `manifests/`: canonical and shard manifests copied from the Drive-backed publication run.
- `final_statistics/`: Stage-15 experiment ledger, confidence intervals, paired significance tests, claim-evidence matrix, per-attack binary metrics, complexity metrics, and environment capture.

## Acceptance boundary

- Stages 05-15 have completed canonical manifests.
- Stage 10 contains 162 registered runs.
- Stage 11 contains 27 registered runs.
- Stage 12 v2 contains 180 registered runs and preserves the documented CPU-to-CUDA resume provenance.
- Stage 13 contains three leave-one-dataset-out folds.
- Stage 14 contains 18 matched policy/resource runs.
- Stage 15 records 447 experiment-ledger rows, 225 confidence intervals, 12 paired tests, 870 per-attack rows, 15 complexity rows, and 2,394 checksum entries.

The full checkpoint/model/result hierarchy is not committed. Completion must be interpreted together with the manifest status, expected run-key coverage, and the documented artifact-count audits.

## Integrity notes

- Failed Stage-12 v1 evidence is preserved in the canonical research archive and is not scored as accepted evidence.
- Energy remains labeled `model_estimate` unless a direct hardware counter was available for an uninterrupted run.
- Per-attack records are binary detection metrics grouped by original attack labels; they are not multiclass predictions.
- Null and non-significant comparisons are retained.
