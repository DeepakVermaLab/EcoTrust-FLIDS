from __future__ import annotations


def _singleton_safe_batch_size(sample_count: int, requested_batch_size: int) -> int:
    """Choose a nearby batch size whose final batch is not a singleton."""
    if sample_count < 2:
        raise ValueError("Local training requires at least two samples for BatchNorm")
    if requested_batch_size < 2:
        raise ValueError("batch_size must be at least 2 for BatchNorm")

    effective_batch_size = min(int(requested_batch_size), int(sample_count))
    while effective_batch_size > 2 and sample_count % effective_batch_size == 1:
        effective_batch_size -= 1
    if sample_count % effective_batch_size == 1:
        effective_batch_size = sample_count
    return effective_batch_size


def train_local_model(
    model,
    X,
    y,
    epochs: int = 1,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
    global_state: dict | None = None,
    proximal_mu: float = 0.0,
):
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("Install project dependencies first: python -m pip install -r requirements.txt") from exc

    if proximal_mu < 0:
        raise ValueError("proximal_mu must be non-negative")
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y), dtype=torch.long)
    effective_batch_size = _singleton_safe_batch_size(len(y_t), batch_size)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=effective_batch_size,
        shuffle=True,
        generator=generator,
    )
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    reference = None
    if proximal_mu > 0:
        if global_state is None:
            raise ValueError("global_state is required when proximal_mu is positive")
        reference = {
            name: tensor.detach().to(device)
            for name, tensor in global_state.items()
            if hasattr(tensor, "is_floating_point") and tensor.is_floating_point()
        }
    total_loss = 0.0
    total = 0
    correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            logits = output[0] if isinstance(output, tuple) else output
            loss = loss_fn(logits, yb)
            if reference is not None:
                proximal = torch.zeros((), device=device)
                for name, parameter in model.named_parameters():
                    if name in reference:
                        proximal = proximal + torch.sum((parameter - reference[name]) ** 2)
                loss = loss + 0.5 * proximal_mu * proximal
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(xb)
            total += len(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().detach().cpu())
            true_positive += int(((pred == 1) & (yb == 1)).sum().detach().cpu())
            false_positive += int(((pred == 1) & (yb == 0)).sum().detach().cpu())
            false_negative += int(((pred == 0) & (yb == 1)).sum().detach().cpu())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return model.state_dict(), {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n": total,
    }
