from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import StockLSTM

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "stock_lstm.pt"

DEVICE = torch.device("cpu")
LEARNING_RATE = 0.001
NUM_EPOCHS = 20


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


def main() -> None:
    train_loader, _ = get_dataloaders()

    model = StockLSTM().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Egitim baslatiliyor: {NUM_EPOCHS} epoch, cihaz: {DEVICE}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}] - train loss: {train_loss:.6f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nEgitilen model '{MODEL_PATH}' olarak kaydedildi.")


if __name__ == "__main__":
    main()
