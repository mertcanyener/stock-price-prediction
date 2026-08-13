import pickle
from pathlib import Path

import numpy as np
import torch

from model import StockLSTM

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# prepare_dataset.py'daki FEATURE_COLUMNS ile ayni sira: Close scaler'in ilk kolonu.
NUM_FEATURES = 5
CLOSE_INDEX = 0

DEVICE = torch.device("cpu")


def inverse_transform_close(values: np.ndarray, scaler) -> np.ndarray:
    """values: (N,) olceklenmis Close degerleri. Scaler 5 kolonla fit edildigi
    icin, geri donusum yapabilmek icin 5 kolonluk bir dummy array olusturup
    Close degerini ilk kolona yerlestiriyoruz; diger kolonlar bu donusumde
    kullanilmiyor (MinMaxScaler her kolonu bagimsiz olcekliyor)."""
    dummy = np.zeros((len(values), NUM_FEATURES))
    dummy[:, CLOSE_INDEX] = values
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, CLOSE_INDEX]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape


def predict_test_set(ticker: str) -> tuple[np.ndarray, np.ndarray]:
    """Test seti uzerinde tahmin yapar, gerceklestirdigi Close degerleriyle
    birlikte gercek dolar olcegindeki (y_true, y_pred) ciftini dondurur."""
    scaler_path = DATA_DIR / f"{ticker}_scaler.pkl"
    model_path = MODELS_DIR / f"{ticker}_stock_lstm.pt"

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = StockLSTM().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    X_test = np.load(DATA_DIR / f"{ticker}_X_test.npy")
    y_test = np.load(DATA_DIR / f"{ticker}_y_test.npy")

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).squeeze(1).numpy()

    y_pred = inverse_transform_close(y_pred_scaled, scaler)
    y_true = inverse_transform_close(y_test, scaler)

    return y_true, y_pred


def main(ticker: str = "AAPL") -> None:
    y_true, y_pred = predict_test_set(ticker)
    rmse, mae, mape = compute_metrics(y_true, y_pred)

    print(f"[{ticker}] Test seti ornek sayisi: {len(y_true)}")
    print(f"[{ticker}] RMSE: ${rmse:.2f}")
    print(f"[{ticker}] MAE:  ${mae:.2f}")
    print(f"[{ticker}] MAPE: {mape:.2f}%")


if __name__ == "__main__":
    main()
