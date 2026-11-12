# EcoTrust-FLIDS Literature Map

This map organizes the verified 48-paper set for problem-history and research-gap analysis. Paper IDs refer to `literature_manifest.csv`. It is an evidence-routing aid, not a substitute for claim-by-claim full-text review.

## Historical progression, 2023-2026

- **2023 — feasibility and foundational defenses:** early work establishes federated IDS feasibility across IIoT, IoV, FinTech, and lightweight IoT settings (`01`-`05`), while surveys and methods expose privacy, poisoning, Byzantine aggregation, and malicious-client incentives as unresolved security concerns (`06`-`10`).
- **2024 — domain expansion and architectural diversity:** studies add IoMT/blockchain, cyber-physical systems, SDN, vehicle networks, split learning, transfer learning, secure aggregation, differential privacy, and group-based personalization (`11`-`28`). This period shows that one FL-IDS configuration does not automatically generalize across domains, privacy mechanisms, or adversarial conditions.
- **2025 — realism and evaluation gaps:** the literature increasingly targets rare attacks, non-IID data, knowledge distillation, heterogeneity, explainability, semi-supervision, graph learning, and multi-scale evaluation (`25`, `29`-`44`). These papers motivate evaluation beyond a single IID accuracy number.
- **2026 — integrated robustness and imbalance:** current work addresses multi-domain class imbalance, poisoning-resistant vehicle security, mixture-of-experts IDS, privacy-aware oversampling, and Byzantine-robust UAV security (`43`, `45`-`48`).

## Evidence clusters

| Problem cluster | Representative paper IDs | Connection to EcoTrust-FLIDS evaluation |
| --- | --- | --- |
| Core federated IDS feasibility | `01`, `02`, `05`, `11`, `13`, `16`, `19`, `29`, `32`, `39`, `40`, `44`, `46` | Justifies matched federated baselines and multi-dataset evaluation. |
| Non-IID, heterogeneity, rare attacks, imbalance | `25`, `30`, `31`, `33`, `36`, `37`, `43`, `47` | Motivates Stage 10 heterogeneity/scalability tests and per-class/rare-attack reporting. |
| Poisoning, Byzantine clients, and trust | `07`-`10`, `24`, `27`, `35`, `45`, `48` | Motivates Stage 12 malicious-client experiments and trust-aware aggregation comparisons. |
| Privacy and secure collaboration | `03`, `12`, `14`, `17`, `18`, `22`, `23`, `30`, `42`, `47` | Requires explicit privacy/utility accounting rather than treating FL alone as a privacy guarantee. |
| Personalization, hierarchy, and distributed control | `13`, `19`, `20`, `23`, `28`, `37`, `46` | Supports controller and cross-domain policy comparisons while highlighting architecture-specific assumptions. |
| Explainability and adaptive representation | `06`, `19`, `31`, `34`, `38`, `41`, `46`, `48` | Supports policy transparency, rare-class analysis, and evidence beyond aggregate accuracy. |
| Efficiency and deployment constraints | `05`, `19`, `25`, `30`, `32`, `37`, `44` | Motivates measured runtime, memory, communication, controller overhead, and explicitly labeled energy estimates. |

## Problem statement supported by the set

The literature establishes that federated intrusion detection is technically viable, but evidence is fragmented across datasets, domains, aggregation rules, privacy mechanisms, and threat models. Many studies optimize one axis—accuracy, privacy, non-IID learning, robustness, or efficiency—without evaluating all of the following under one matched protocol:

1. multiple modern IoT/IIoT intrusion datasets;
2. multiple seeds with uncertainty and significance testing;
3. non-IID severity, client dropout, stragglers, and scalability;
4. temporal drift and catastrophic forgetting;
5. malicious-client attacks and robust aggregation;
6. rare-attack and per-class performance;
7. runtime, memory, communication, controller overhead, and clearly labeled energy evidence;
8. cross-dataset controller transfer and reproducibility artifacts.

EcoTrust-FLIDS is therefore being evaluated as a unified evidence framework for trust-aware, incremental, resource-conscious federated intrusion detection—not merely as another single-dataset classifier. This framing remains provisional until the remaining registered experiments and final Stage-15 statistical audit are complete. The literature set alone does not establish superiority or journal readiness.

## Required next literature step

Before manuscript claims are written, extract a structured claim matrix from the full texts: datasets, client count, partition model, seeds, baselines, threat model, drift protocol, per-class metrics, calibration, resource metrics, statistical tests, code/data availability, limitations, and exact reported results. Any claim of novelty must be checked against that matrix rather than inferred from titles or abstracts.
