from pathlib import Path

import matplotlib.pyplot as plt

from evaluate import predict_test_set
from prepare_dataset import SEQUENCE_LENGTH, load_data, split_train_test

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def get_test_dates(ticker: str):
    df = load_data(ticker)
    _, test_df = split_train_test(df, ticker)
    # create_sequences ilk SEQUENCE_LENGTH satiri gecmis pencere olarak kullanir;
    # ilk tahmin edilen gun test_df.index[SEQUENCE_LENGTH]'e karsilik gelir.
    return test_df.index[SEQUENCE_LENGTH:]


def plot_predictions(ticker: str) -> Path:
    y_true, y_pred = predict_test_set(ticker)
    dates = get_test_dates(ticker)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label="Gerçek", color="tab:blue")
    plt.plot(dates, y_pred, label="Tahmin", color="tab:orange", linestyle="--")

    plt.title(f"{ticker} Close Fiyat Tahmini - Gerçek vs Tahmin")
    plt.xlabel("Tarih")
    plt.ylabel("Fiyat (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"{ticker}_prediction_vs_actual.png"
    plt.savefig(output_path)
    plt.close()

    print(f"[{ticker}] Grafik '{output_path}' olarak kaydedildi.")
    return output_path


def main() -> None:
    plot_predictions("AAPL")


if __name__ == "__main__":
    main()
