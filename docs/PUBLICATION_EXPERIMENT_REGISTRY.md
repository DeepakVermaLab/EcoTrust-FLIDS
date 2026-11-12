# Publication Experiment Registry

`configs/publication_experiments.json` is frozen before stage 09. It is the
machine-readable protocol for the remaining publication experiments.

## Locked Rules

- Reuse the stage-02 train/test split for matched comparisons.
- Derive validation data from training data only.
- Do not tune on the test split.
- Run every matched neural method from the same stage-04 initialization.
- Use the registered primary seeds for all main comparisons.
- Preserve failed runs and negative results in stage manifests and the final
  experiment ledger.
- Report estimated energy as estimated. Direct energy claims require a recorded
  RAPL or NVML measurement source.
- Do not call a stage complete from notebook output alone. Validate its manifest
  and registered artifacts on Drive.

## Scope Decisions

Cross-dataset evaluation targets controller generalization rather than direct
feature-level IDS transfer. The three datasets have different input schemas and
separately trained encoders, so treating their latent coordinates as directly
interchangeable would not be a defensible transfer experiment.

Temporal evaluation uses chronological ordering only when a defensible time
field exists. When it does not, acquisition/source order is reported explicitly
as source-order drift and is not described as natural temporal drift.

The registered matrix separates primary comparisons from sensitivity studies
so the full protocol remains executable on CPU while still testing multiple
seeds, non-IID severity, dropout, scale, drift, malicious clients, controller
generalization, calibration, and resource cost.

## Stage 09 Protocol Amendment

The first FedAdam run exposed a numerical implementation defect before any
valid FedAdam round metric was produced: the server update used an implicit
learning rate of 1.0 and generated non-finite probabilities. On 2026-10-23 the
FedAdam-only server configuration was made explicit with learning rate 0.01,
beta1 0.9, beta2 0.99, and tau 0.001. The configuration is now stored in every
FedAdam checkpoint and non-finite parameters or probabilities fail explicitly.
The six completed FedAvg and FedProx runs are unchanged. This repair was not
selected from test-set performance and the original failed manifest is
preserved.

## Stage 10 Operational Definitions

Before stage 10 execution, its registered sensitivity protocol was clarified
without inspecting any stage-10 results. FedAvg and EcoTrust are compared under
all primary seeds. The one-factor-at-a-time matrix is deduplicated at the
primary setting and augmented by the hardest joint condition. A dropout occurs
before local training and therefore consumes neither training energy nor
communication. A straggler completes local training, consumes compute energy,
but misses the aggregation deadline, so its update is excluded.

Stage 10 may be executed as two deterministic run shards to meet the authorized
execution deadline. Sharding changes only scheduling: every registered dataset,
scenario, method, seed, round, and metric remains unchanged. Each shard writes a
separate manifest. The canonical stage manifest is created atomically only when
both shard manifests are complete, their run-key digests match, no run key is
duplicated, and their union exactly covers all 162 registered runs.

## Stage 11 Operational Definitions

Before stage 11 execution and without inspecting stage-11 results, the stream
comparison was fixed to naive sequential FedAvg, reservoir replay, and EcoTrust
with class-balanced replay. All methods use the primary seeds, five ordered
training windows, and the same stage-04 initialization. Each window reserves a
training-only validation subset for forgetting calculations. The locked test
split is evaluated once after the final stream window and is never used for
adaptation or selection. Stream order is chronological only when a timestamp
passes validity checks; otherwise it is explicitly reported as acquisition or
source order.

## Stage 12 Operational Definitions

The full malicious-client attack, fraction, and aggregation matrix is a
deterministic sensitivity study using seed 42. Malicious clients are sampled
independently of their trust score. Label flipping changes only malicious local
labels; sign flip, model scaling, Gaussian update, and free-rider behavior are
applied to model deltas after otherwise matched local training. Stage 09 remains
the registered multi-seed source for primary inferential comparisons.

The active Stage-12 poisoning protocol is `parameter_updates_only_v2` (amended
2026-11-01 before any canonical Stage-12 result existed). Sign flip, model
scaling, and Gaussian attacks modify trainable parameter deltas only. BatchNorm
running statistics and other buffers retain their values from matched local
training. The partial v1 implementation also transformed floating BatchNorm
buffers, which could create negative running variance and non-finite
predictions. Those failures remain preserved as invalid v1 evidence and are not
used for claims. The complete 180-run matrix must be rerun under v2 in its
versioned output namespace.

Stage 12 may be executed as three deterministic run shards. Each shard contains
60 of the 180 registered runs and writes an independent manifest. The canonical
manifest is created only after exact run-key coverage, digest, completion, and
duplicate checks pass. Shards with a different shard count must never be mixed.

On 2026-11-02, after all three v2 shards had persisted 37/60 runs and repeated
Colab CPU-runtime disconnections, the user authorized a GPU continuation. This
is an execution-backend amendment, not a change to the attack protocol or model
comparison. The 111 completed CPU runs remain valid and are retained. Each
interrupted checkpoint may resume on CUDA, but the local batch size remains 256
to avoid changing both hardware and optimization batching at the transition.
Checkpoints, per-round rows, summaries, and manifests record device history and
the `cpu_to_gpu_resume_v1` amendment. All remaining shards must use the same GPU
class; if three matching GPUs are unavailable, execute the shards sequentially
on the same GPU class. Mixed CPU/GPU provenance must be disclosed and must not
be interpreted as a GPU-versus-CPU performance comparison.

## Stage 13 Operational Definitions

Cross-dataset evaluation concerns controller decisions, not direct IDS feature
transfer. Each leave-one-dataset-out fold trains the learned policy only on
stage-06 client telemetry from two datasets and evaluates it without refitting
on the third. Learned inputs are limited to trust, drift, normalized remaining
energy, and normalized round index. The reference label is the deterministic
stage-06 telemetry-agent decision. Random-budget, round-robin, static-trust,
telemetry-heuristic, and learned policies use the same 60 percent budget.
Remaining energy is normalized by each client's recorded initial energy budget,
not by a shared nominal constant.

## Stage 14 Operational Definitions

The matched policy resource comparison uses seed 42 and the same stage-04
initialization, clients, rounds, epochs, and trust-weighted aggregation for all
six policies. It records measured wall time, process CPU time, peak RSS,
communication, and controller decision overhead. Energy is labeled direct only
when a readable RAPL or NVML counter is present; otherwise the calibrated
simulator output is reported explicitly as an estimate. The learned controller
for each dataset is trained only on stage-06 telemetry from the other two.

## Stage 15 Completion Rules

Stage 15 is a reporting and evidence-audit stage, not another tuning stage. It
refuses to run unless the stage-09 through stage-14 manifests all report
`completed`. It aggregates every recorded run, computes deterministic
multi-seed confidence intervals and paired tests, and applies Holm correction
without removing negative comparisons. Per-attack rows retain the original
attack labels for the locked stage-02 test membership, but they report binary
detection recall and false-negative rate; they are not multiclass
classification results. Normal traffic is reported with false-positive rate
and specificity. Complexity rows record trainable parameters, model-state
bytes, model-file bytes, and linear-layer multiply-accumulate counts.

The final checksum inventory covers the frozen registry, project source,
notebooks, documentation, stage manifests, and publication result artifacts.
The stage-15 manifest is marked complete only after every registered artifact
exists and is nonempty.
