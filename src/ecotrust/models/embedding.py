from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _torch_deps():
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("Install project dependencies first: python -m pip install -r requirements.txt") from exc
    return np, torch, nn, DataLoader, TensorDataset


@dataclass
class EmbeddingTrainConfig:
    input_dim: int
    embedding_dim: int = 64
    hidden_dim: int = 128
    epochs: int = 5
    batch_size: int = 512
    lr: float = 1e-3
    device: str = "cpu"
    trust_weight: float = 0.15
    reconstruction_weight: float = 0.35


def build_encoder_class():
    _, torch, nn, _, _ = _torch_deps()

    class EcoTrustEncoder(nn.Module):
        def __init__(self, input_dim: int, embedding_dim: int = 64, hidden_dim: int = 128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, embedding_dim),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(embedding_dim, 2)
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            logits = self.classifier(z)
            recon = self.decoder(z)
            return logits, z, recon

    return EcoTrustEncoder


def train_embedding_model(X_train, y_train, output_dir: str | Path, cfg: EmbeddingTrainConfig):
    np, torch, nn, DataLoader, TensorDataset = _torch_deps()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X = torch.tensor(np.asarray(X_train), dtype=torch.float32)
    y = torch.tensor(np.asarray(y_train), dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model_cls = build_encoder_class()
    model = model_cls(cfg.input_dim, cfg.embedding_dim, cfg.hidden_dim).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    history = []
    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits, z, recon = model(xb)
            compactness = z.pow(2).mean()
            loss = ce(logits, yb) + cfg.reconstruction_weight * mse(recon, xb) + cfg.trust_weight * compactness
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(xb)
            total += len(xb)
            correct += int((logits.argmax(dim=1) == yb).sum().detach().cpu())
        history.append({"epoch": epoch + 1, "loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)})

    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__, "history": history}, out / "ecotrust_encoder.pt")
    return model, history


def load_embedding_model(checkpoint_path: str | Path, device: str = "cpu"):
    """Restore the trained stage-04 encoder for federated fine-tuning."""
    _, torch, _, _, _ = _torch_deps()
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Embedding checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    config = checkpoint.get("config", {})
    required = {"input_dim", "embedding_dim", "hidden_dim"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Embedding checkpoint is missing config keys: {sorted(missing)}")

    model_cls = build_encoder_class()
    model = model_cls(
        int(config["input_dim"]),
        int(config["embedding_dim"]),
        int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def encode_features(model, X, device: str = "cpu"):
    np, torch, _, _, _ = _torch_deps()
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(np.asarray(X), dtype=torch.float32).to(device)
        _, z, _ = model(tensor)
    return z.detach().cpu().numpy()
