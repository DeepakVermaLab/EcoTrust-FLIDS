from __future__ import annotations

import argparse
from pathlib import Path

from ecotrust.config import ensure_project_dirs, load_config, run_mode_settings
from ecotrust.data.audit import audit_datasets
from ecotrust.data.clients import dirichlet_client_splits, save_splits
from ecotrust.data.loaders import load_dataset_frame
from ecotrust.data.preprocess import load_preprocessed, preprocess_frame
from ecotrust.io import write_json


def cmd_audit(args) -> None:
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    out = cfg.results_root / "dataset_audit.json"
    audit_datasets(cfg, out)
    print(f"Wrote {out}")


def cmd_preprocess(args) -> None:
    cfg = load_config(args.config)
    settings = run_mode_settings(cfg, args.mode)
    nrows = settings["rows_per_dataset"]
    frame = load_dataset_frame(args.dataset, cfg, nrows=nrows, max_sources=1 if args.mode == "smoke" else None)
    out = cfg.results_root / args.dataset / "preprocessed" / args.mode
    result = preprocess_frame(frame, cfg.datasets[args.dataset], out, random_state=cfg.seed)
    splits = dirichlet_client_splits(result.y_train, n_clients=settings["max_clients"], seed=cfg.seed)
    save_splits(splits, out / "client_splits.json")
    write_json(
        out / "preprocess_manifest.json",
        {
            "dataset": args.dataset,
            "mode": args.mode,
            "rows_loaded": len(frame),
            "label_column": result.label_column,
            "attack_type_column": result.attack_type_column,
            "n_features": len(result.feature_names),
            "n_clients": len(splits),
        },
    )
    print(f"Wrote preprocessing artifacts to {out}")


def cmd_baseline(args) -> None:
    from ecotrust.evaluation.reports import write_metrics_report
    from ecotrust.models.baselines import train_baselines

    cfg = load_config(args.config)
    prep = cfg.results_root / args.dataset / "preprocessed" / args.mode
    X_train, X_test, y_train, y_test = load_preprocessed(prep)
    out = cfg.results_root / args.dataset / "baselines" / args.mode
    metrics = train_baselines(X_train, y_train, X_test, y_test, out, seed=cfg.seed)
    write_metrics_report(out, f"{args.dataset} baseline models", metrics)
    print(f"Wrote baseline artifacts to {out}")


def cmd_embedding(args) -> None:
    import pandas as pd

    from ecotrust.models.embedding import EmbeddingTrainConfig, encode_features, train_embedding_model

    cfg = load_config(args.config)
    prep = cfg.results_root / args.dataset / "preprocessed" / args.mode
    X_train, X_test, y_train, _ = load_preprocessed(prep)
    out = cfg.results_root / args.dataset / "embedding" / args.mode
    train_cfg = EmbeddingTrainConfig(
        input_dim=X_train.shape[1],
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        device=args.device,
    )
    model, history = train_embedding_model(X_train, y_train, out, train_cfg)
    z_train = encode_features(model, X_train, device=args.device)
    z_test = encode_features(model, X_test, device=args.device)
    pd.DataFrame(z_train).to_parquet(out / "Z_train.parquet", index=False)
    pd.DataFrame(z_test).to_parquet(out / "Z_test.parquet", index=False)
    write_json(out / "embedding_history.json", history)
    print(f"Wrote embedding artifacts to {out}")


def cmd_fed(args) -> None:
    import json

    from ecotrust.data.clients import ClientSplit
    from ecotrust.federated.incremental import run_incremental_federated_training
    from ecotrust.models.embedding import EmbeddingTrainConfig, train_embedding_model

    cfg = load_config(args.config)
    prep = cfg.results_root / args.dataset / "preprocessed" / args.mode
    X_train, X_test, y_train, y_test = load_preprocessed(prep)
    with (prep / "client_splits.json").open("r", encoding="utf-8") as fh:
        splits = [ClientSplit(**item) for item in json.load(fh)]
    settings = run_mode_settings(cfg, args.mode)
    model, _ = train_embedding_model(
        X_train,
        y_train,
        cfg.results_root / args.dataset / "federated" / args.mode / "initial_encoder",
        EmbeddingTrainConfig(input_dim=X_train.shape[1], epochs=1, device=args.device),
    )
    out = cfg.results_root / args.dataset / "federated" / args.mode
    result = run_incremental_federated_training(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        splits,
        out,
        rounds=settings["rounds"],
        local_epochs=settings["local_epochs"],
        device=args.device,
    )
    write_json(out / "federated_summary.json", result)
    print(f"Wrote federated artifacts to {out}")


def cmd_agent(args) -> None:
    import pandas as pd

    from ecotrust.agent.controller import ClientState, EcoTrustAgent
    from ecotrust.energy.simulator import energy_saving_percent, estimate_round_energy

    cfg = load_config(args.config)
    out = cfg.results_root / args.dataset / "agent_energy" / args.mode
    out.mkdir(parents=True, exist_ok=True)
    states = [
        ClientState(f"client_{i:02d}", 0.5 + i * 0.05, 600 + i * 80, 0.2, 0.82 + i * 0.01, (i % 4) / 3)
        for i in range(10)
    ]
    decisions = EcoTrustAgent().decide(states, budget_fraction=0.6)
    rows = []
    baseline = 0.0
    proposed = 0.0
    state_map = {state.client_id: state for state in states}
    for decision in decisions:
        state = state_map[decision.client_id]
        all_selected = estimate_round_energy(5000, 0.4, True)
        agent_selected = estimate_round_energy(5000, 0.4, decision.selected)
        baseline += all_selected["total_j"]
        proposed += agent_selected["total_j"]
        rows.append({**state.__dict__, **decision.__dict__, "agent_energy_j": agent_selected["total_j"]})
    pd.DataFrame(rows).to_csv(out / "agent_decisions.csv", index=False)
    write_json(
        out / "energy_summary.json",
        {"baseline_energy_j": baseline, "proposed_energy_j": proposed, "saving_percent": energy_saving_percent(baseline, proposed)},
    )
    print(f"Wrote agent-energy artifacts to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecotrust", description="EcoTrust-FLIDS experiment runner")
    parser.add_argument("--config", default=None, help="Path to configs/project.json")
    sub = parser.add_subparsers(required=True)

    audit = sub.add_parser("audit")
    audit.set_defaults(func=cmd_audit)

    for name, fn in [
        ("preprocess", cmd_preprocess),
        ("baseline", cmd_baseline),
        ("embedding", cmd_embedding),
        ("fed", cmd_fed),
        ("agent", cmd_agent),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, choices=["Edge-IIoTset", "TON_IoT", "CICIoT2023"])
        p.add_argument("--mode", default="smoke", choices=["smoke", "full"])
        p.set_defaults(func=fn)
        if name in {"embedding", "fed"}:
            p.add_argument("--device", default="cpu")
        if name == "embedding":
            p.add_argument("--epochs", type=int, default=5)
            p.add_argument("--embedding-dim", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
