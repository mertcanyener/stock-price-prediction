from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BATCH_SIZE = 32


class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        # (N,) -> (N, 1) so targets line up with the model's (batch, 1) output.
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_dataloaders(batch_size: int = BATCH_SIZE) -> tuple[DataLoader, DataLoader]:
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()

    print(f"Train dataset boyutu: {len(train_loader.dataset)}")
    print(f"Test dataset boyutu:  {len(test_loader.dataset)}\n")

    X_batch, y_batch = next(iter(train_loader))
    print(f"Train batch -> X shape: {tuple(X_batch.shape)}, y shape: {tuple(y_batch.shape)}")
    assert X_batch.shape == (BATCH_SIZE, 30, 5), f"Beklenmeyen X_batch shape: {tuple(X_batch.shape)}"
    assert y_batch.shape == (BATCH_SIZE, 1), f"Beklenmeyen y_batch shape: {tuple(y_batch.shape)}"

    X_batch, y_batch = next(iter(test_loader))
    print(f"Test batch  -> X shape: {tuple(X_batch.shape)}, y shape: {tuple(y_batch.shape)}")
    assert X_batch.shape[1:] == (30, 5), f"Beklenmeyen X_batch shape: {tuple(X_batch.shape)}"
    assert y_batch.shape[1] == 1, f"Beklenmeyen y_batch shape: {tuple(y_batch.shape)}"

    print("\nTrain ve test DataLoader batch shape'leri dogrulandi.")
