# EcoTrust-FLIDS Research Demonstrator

This directory contains a static, evidence-backed demonstrator for the completed EcoTrust-FLIDS implementation. It is designed for local review, paper screenshots, and later GitHub Pages deployment.

## Scientific boundary

- Accepted Stage 14 policy endpoints and Stage 15 statistical outputs are displayed as recorded evidence.
- Interactive client-selection decisions reproduce the deterministic EcoTrust controller formula.
- Custom budget, drift, trust, energy, and latency settings are labelled as a **what-if simulation**.
- Energy shown for custom scenarios is model-estimated. It is not direct hardware-energy measurement.
- The interface does not perform live intrusion inference and does not create new experimental evidence.
- Binary per-attack detection evidence is not presented as multiclass attack classification.

## Structure

```text
deployment/
|-- evidence/               # exact downloaded accepted artifacts
|-- scripts/build_site_data.py
|-- dist/                   # deployable static site
|   |-- index.html
|   |-- styles.css
|   |-- app.js
|   |-- data.js             # generated from accepted artifacts
|   `-- downloads/          # compact downloadable evidence
|-- screenshots/            # paper-ready browser captures
`-- README.md
```

## Rebuild the evidence bundle

```bash
cd "/Volumes/LocalDrive1/onlyscholar-projects/EcoTrust-FLIDS[DeepakSir]"
python3 IMPLEMENTATION/deployment/scripts/build_site_data.py
```

The compiler validates completed Stage 09-15 manifests, reads Stage 14 policy summaries and round traces, parses Stage 15 statistics, copies a compact download bundle, and writes SHA-256 provenance for every downloadable file.

## Local preview

```bash
cd "/Volumes/LocalDrive1/onlyscholar-projects/EcoTrust-FLIDS[DeepakSir]/IMPLEMENTATION/deployment/dist"
python3 -m http.server 4174 --bind 127.0.0.1
```

Open `http://127.0.0.1:4174/`.

## GitHub Pages

The `dist/` directory is self-contained and uses only relative paths. It can be deployed with GitHub Actions, `peaceiris/actions-gh-pages`, or any static host. No server-side runtime, API key, model checkpoint, or dataset is required.

