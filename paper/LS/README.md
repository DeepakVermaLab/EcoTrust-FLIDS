# EcoTrust-FLIDS Literature Set

This directory is reserved for the verified 2023-2026 journal literature set supporting the EcoTrust-FLIDS second paper.

Inclusion requirements:

- final journal publication dated 2023, 2024, 2025, or 2026;
- DOI and publisher/Crossref metadata available;
- relevant to federated intrusion detection, IoT/IIoT security, non-IID learning, malicious-client robustness, continual drift/forgetting, client selection, energy/resource efficiency, or cross-dataset evaluation;
- journal indexed in Scopus and/or Web of Science at the journal level, with the verification basis recorded in the manifest;
- lawful full-text PDF from a publisher or institutional repository;
- no arXiv-only, conference-only, retracted, or early-release-only record.

`ref.bib` is populated from DOI registry/publisher metadata. `literature_manifest.csv` is the audit ledger for all downloaded papers.

## Verified contents

- `papers/`: 48 readable PDF files, each downloaded from a publisher or institutional repository and checked for PDF signature, page count, title agreement, and SHA-256.
- `ref.bib`: 48 unique `@article` records sourced from the Crossref DOI content-negotiation endpoint; entries were not written from memory. Canonical DOI URLs, BibTeX-safe ampersands, and five locators omitted by Crossref's BibTeX transform were normalized from the corresponding Crossref JSON records.
- `literature_manifest.csv`: DOI, final issue year, title, journal, article locator, topic, PDF source, file size, page count, checksum, and metadata source for every paper.
- `reference_mapping.csv`: one-to-one mapping from every downloaded PDF and BibTeX key to its authors, year, title, journal, volume/issue/locator, DOI, canonical DOI link, publisher link, PDF source, Crossref source, saved metadata, and SHA-256. All 48 rows were live-verified against Crossref and doi.org.
- `indexing_verification.csv`: ISSN-level match evidence against Elsevier's official September 2026 Scopus source-title workbook, plus an informational match against August 2026 SCIE/ESCI snapshots whose provenance points to Clarivate's official Master Journal List downloads page.
- `selected_dois.tsv`: the exact 48-record inclusion ledger.
- `metadata/`: saved Crossref and OpenAlex source responses for auditability.
- `build_literature_set.py` and `verify_indexing.py`: reproducible acquisition and verification scripts.
- `verify_reference_mapping.py`: validates local PDF/BibTeX/metadata agreement and, with `--live`, rechecks Crossref plus doi.org before regenerating `reference_mapping.csv`.

The final year distribution is 10 papers from 2023, 17 from 2024, 16 from 2025, and 5 from 2026. All 48 journal sources match the official September 2026 Scopus source list and the selected article year falls inside the listed coverage. The WoS snapshot match is 33 SCIE, 12 ESCI, and 3 Scopus-only journals. The September 2026 Scopus list labels the Indonesian Journal of Electrical Engineering and Computer Science inactive after coverage through 2025; the included article is dated 2025 and therefore remains inside the verified Scopus coverage period.

No downloaded source URL is from arXiv, ResearchGate, Academia.edu, Preprints.org, TechRxiv, or Sci-Hub. Reviews were retained only where they help establish the historical taxonomy, attack surface, or explicit research gaps; the set also includes implementation/evaluation studies for non-IID data, rare attacks, poisoning and Byzantine clients, drift-adjacent adaptation, privacy, explainability, efficiency, and multi-domain IDS evaluation.

## Rebuild and verify

Use the bundled workspace Python because it provides `pypdf` and `openpyxl`:

```bash
cd '/Volumes/LocalDrive1/onlyscholar-projects/EcoTrust-FLIDS[DeepakSir]'
/Users/darsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  IMPLEMENTATION/paper/LS/build_literature_set.py
/Users/darsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  IMPLEMENTATION/paper/LS/verify_indexing.py
/Users/darsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  IMPLEMENTATION/paper/LS/verify_reference_mapping.py --live
```
