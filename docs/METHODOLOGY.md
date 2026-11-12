# EcoTrust-FLIDS Methodology

## Research Positioning

EcoTrust-FLIDS is designed as an incremental second-paper extension over the first energy-aware FLIDS work. The contribution is not only a larger classifier. The implementation isolates four paper-grade components:

1. Adaptive latent feature encoder for compact intrusion representations.
2. Trust-weighted incremental federated learning for client/domain drift.
3. Agentic 24x7 controller that selects, sleeps, or delays clients using model telemetry, trust, drift, and energy state.
4. Energy accounting and ablation reports that compare all-client FL against agent-controlled FL.

## Experimental Flow

1. Audit downloaded source files and labels.
2. Preprocess each dataset into train/test vectors.
3. Create non-IID federated client splits.
4. Train classical baselines.
5. Train the EcoTrust adaptive encoder and export latent vectors.
6. Run the all-client, trust-weighted federated comparator from the trained encoder.
7. Run the agent-selected counterpart with the same initialization and training budget.
8. Quantify energy saving, performance retention, and ablation deltas.

## Publication Evidence Required

For IEEE Access-level submission, the final paper should include:

- Per-dataset binary and attack-type class distributions.
- Baseline comparison against first-paper model family and current FLIDS baselines.
- Ablations: no embedding, no trust, no agent, no incremental replay, all-client FL.
- Non-IID sensitivity using multiple Dirichlet alpha values.
- Cross-dataset drift or leave-one-dataset-out generalization.
- Energy calculation assumptions and sensitivity analysis.
- Repeated seeds with confidence intervals.

The current code creates the pipeline needed for these outputs. The final claim strength depends on the full-run results, not on the scaffold itself.
