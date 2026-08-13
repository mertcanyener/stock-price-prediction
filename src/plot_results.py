from pathlib import Path

import matplotlib.pyplot as plt

from evaluate import predict_test_set
from prepare_dataset import SEQUENCE_LENGTH, load_data, split_train_test

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_PATH = OUTPUTS_DIR / "prediction_vs_actual.png"


def get_test_dates():
    df = load_data()
    _, test_df = split_train_test(df)
    # create_sequences ilk SEQUENCE_LENGTH satiri gecmis pencere olarak kullanir;
    # ilk tahmin edilen gun test_df.index[SEQUENCE_LENGTH]'e karsilik gelir.
    return test_df.index[SEQUENCE_LENGTH:]


def main() -> None:
    y_true, y_pred = predict_test_set()
    dates = get_test_dates()

    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label="Gerçek", color="tab:blue")
    plt.plot(dates, y_pred, label="Tahmin", color="tab:orange", linestyle="--")

    plt.title("AAPL Close Fiyat Tahmini - Gerçek vs Tahmin")
    plt.xlabel("Tarih")
    plt.ylabel("Fiyat (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    print(f"Grafik '{OUTPUT_PATH}' olarak kaydedildi.")


if __name__ == "__main__":
    main()
