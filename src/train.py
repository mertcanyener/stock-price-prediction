from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import StockLSTM

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "stock_lstm.pt"

DEVICE = torch.device("cpu")


def train_one_epoch(model: nn.Module, loader, criterion, optimizer) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


def train_model(
    learning_rate: float = 0.001,
    batch_size: int = 32,
    num_epochs: int = 20,
    hidden_size: int = 64,
    verbose: bool = True,
) -> tuple[nn.Module, float]:
    train_loader, _ = get_dataloaders(batch_size=batch_size)

    model = StockLSTM(hidden_size=hidden_size).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    final_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        final_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        if verbose:
            print(f"  Epoch [{epoch:2d}/{num_epochs}] - train loss: {final_loss:.6f}")

    return model, final_loss


def main() -> None:
    experiments = [
        {"name": "lr=0.001, epoch=20, hidden=64 (baseline)", "learning_rate": 0.001, "num_epochs": 20, "hidden_size": 64},
        {"name": "lr=0.0005, epoch=30, hidden=64", "learning_rate": 0.0005, "num_epochs": 30, "hidden_size": 64},
        {"name": "lr=0.001, epoch=30, hidden=32", "learning_rate": 0.001, "num_epochs": 30, "hidden_size": 32},
    ]

    results = []
    best_model = None
    best_loss = float("inf")
    best_name = None

    for config in experiments:
        print(f"\n=== Deneme: {config['name']} ===")
        model, final_loss = train_model(
            learning_rate=config["learning_rate"],
            num_epochs=config["num_epochs"],
            hidden_size=config["hidden_size"],
        )
        results.append((config["name"], final_loss))

        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model
            best_name = config["name"]

    print("\n=== Karsilastirma (final train loss) ===")
    for name, loss in results:
        marker = "  <-- en iyi" if name == best_name else ""
        print(f"  {name}: {loss:.6f}{marker}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), MODEL_PATH)
    print(f"\nEn iyi kombinasyon: {best_name} (final train loss: {best_loss:.6f})")
    print(f"Model '{MODEL_PATH}' olarak kaydedildi.")


if __name__ == "__main__":
    main()
