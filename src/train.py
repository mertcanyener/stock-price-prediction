from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import StockLSTM

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

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
    ticker: str,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    num_epochs: int = 20,
    hidden_size: int = 64,
    verbose: bool = True,
) -> tuple[nn.Module, float]:
    train_loader, _ = get_dataloaders(ticker, batch_size=batch_size)

    model = StockLSTM(hidden_size=hidden_size).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    final_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        final_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        if verbose:
            print(f"  Epoch [{epoch:2d}/{num_epochs}] - train loss: {final_loss:.6f}")

    return model, final_loss


def train_and_save(
    ticker: str,
    learning_rate: float = 0.0005,
    batch_size: int = 32,
    num_epochs: int = 30,
    hidden_size: int = 64,
    verbose: bool = True,
) -> float:
    """Bilinen en iyi hiperparametrelerle tek seferlik egitim yapar ve
    models/{ticker}_stock_lstm.pt olarak kaydeder."""
    model, final_loss = train_model(
        ticker,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        hidden_size=hidden_size,
        verbose=verbose,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{ticker}_stock_lstm.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[{ticker}] Model '{model_path}' olarak kaydedildi (final train loss: {final_loss:.6f}).\n")

    return final_loss


def run_experiments(ticker: str) -> None:
    """Farkli hiperparametre kombinasyonlarini deneyip en iyisini kaydeder
    (tek ticker uzerinde manuel hiperparametre arama icin)."""
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
        print(f"\n=== [{ticker}] Deneme: {config['name']} ===")
        model, final_loss = train_model(
            ticker,
            learning_rate=config["learning_rate"],
            num_epochs=config["num_epochs"],
            hidden_size=config["hidden_size"],
        )
        results.append((config["name"], final_loss))

        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model
            best_name = config["name"]

    print(f"\n=== [{ticker}] Karsilastirma (final train loss) ===")
    for name, loss in results:
        marker = "  <-- en iyi" if name == best_name else ""
        print(f"  {name}: {loss:.6f}{marker}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{ticker}_stock_lstm.pt"
    torch.save(best_model.state_dict(), model_path)
    print(f"\n[{ticker}] En iyi kombinasyon: {best_name} (final train loss: {best_loss:.6f})")
    print(f"Model '{model_path}' olarak kaydedildi.")


def main() -> None:
    run_experiments("AAPL")


if __name__ == "__main__":
    main()
