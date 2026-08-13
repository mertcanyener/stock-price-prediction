import csv
from pathlib import Path

from evaluate import compute_metrics, predict_test_set
from fetch_data import fetch_data
from plot_results import plot_predictions
from prepare_dataset import prepare_dataset
from preprocess import preprocess
from train import train_and_save

TICKERS = ["AAPL", "MSFT", "NVDA", "META", "PLTR", "GOOGL", "TTWO", "TSLA", "AMZN", "AMD"]

BEST_PARAMS = {"learning_rate": 0.0005, "num_epochs": 30, "hidden_size": 64}

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
SUMMARY_PATH = OUTPUTS_DIR / "summary_results.csv"


def run_pipeline_for_ticker(ticker: str) -> dict:
    print(f"\n{'=' * 60}\n{ticker} icin pipeline calistiriliyor\n{'=' * 60}")

    fetch_data(ticker)
    preprocess(ticker)
    prepare_dataset(ticker)
    train_and_save(ticker, **BEST_PARAMS, verbose=False)

    y_true, y_pred = predict_test_set(ticker)
    rmse, mae, mape = compute_metrics(y_true, y_pred)
    plot_predictions(ticker)

    print(f"[{ticker}] RMSE=${rmse:.2f}  MAE=${mae:.2f}  MAPE=%{mape:.2f}")
    return {"ticker": ticker, "rmse": rmse, "mae": mae, "mape": mape}


def print_summary(results: list[dict]) -> None:
    header = f"{'Ticker':<8}{'RMSE ($)':>12}{'MAE ($)':>12}{'MAPE (%)':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['ticker']:<8}{r['rmse']:>12.2f}{r['mae']:>12.2f}{r['mape']:>12.2f}")


def save_summary_csv(results: list[dict]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "rmse", "mae", "mape"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nOzet tablo '{SUMMARY_PATH}' olarak kaydedildi.")


def main() -> None:
    results = [run_pipeline_for_ticker(ticker) for ticker in TICKERS]

    print(f"\n{'=' * 60}\nOzet Sonuclar\n{'=' * 60}")
    print_summary(results)
    save_summary_csv(results)


if __name__ == "__main__":
    main()
