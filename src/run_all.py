import csv
import time
from pathlib import Path

from evaluate import compute_metrics, predict_test_set
from fetch_data import fetch_data
from plot_results import plot_predictions
from prepare_dataset import prepare_dataset
from preprocess import preprocess
from train import train_and_save

TICKERS = ["AAPL", "MSFT", "NVDA", "META", "PLTR", "GOOGL", "TTWO", "TSLA", "AMZN", "AMD"]
MODEL_TYPES = ["lstm", "gru"]

BEST_PARAMS = {"learning_rate": 0.0005, "num_epochs": 30, "hidden_size": 64}

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
SUMMARY_PATH = OUTPUTS_DIR / "summary_results.csv"


def run_pipeline_for_ticker(ticker: str) -> list[dict]:
    print(f"\n{'=' * 60}\n{ticker} icin pipeline calistiriliyor\n{'=' * 60}")

    fetch_data(ticker)
    preprocess(ticker)
    prepare_dataset(ticker)

    results = []
    for model_type in MODEL_TYPES:
        start_time = time.time()
        train_and_save(ticker, model_type=model_type, **BEST_PARAMS, verbose=False)
        training_time = time.time() - start_time

        y_true, y_pred = predict_test_set(ticker, model_type=model_type)
        rmse, mae, mape = compute_metrics(y_true, y_pred)

        print(
            f"[{ticker}][{model_type.upper()}] RMSE=${rmse:.2f}  MAE=${mae:.2f}  "
            f"MAPE=%{mape:.2f}  Egitim suresi={training_time:.1f}s"
        )
        results.append(
            {
                "ticker": ticker,
                "model_type": model_type,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "training_time_seconds": training_time,
            }
        )

    plot_predictions(ticker)

    return results


def print_summary(results: list[dict]) -> None:
    header = f"{'Ticker':<8}{'Model':<8}{'RMSE ($)':>12}{'MAE ($)':>12}{'MAPE (%)':>12}{'Sure (s)':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['ticker']:<8}{r['model_type'].upper():<8}{r['rmse']:>12.2f}"
            f"{r['mae']:>12.2f}{r['mape']:>12.2f}{r['training_time_seconds']:>12.1f}"
        )


def print_model_comparison(results: list[dict]) -> None:
    print(f"\n{'=' * 60}\nLSTM vs GRU Karsilastirmasi (10 hissenin ortalamasi)\n{'=' * 60}")

    averages = {}
    for model_type in MODEL_TYPES:
        model_results = [r for r in results if r["model_type"] == model_type]
        n = len(model_results)
        averages[model_type] = {
            "rmse": sum(r["rmse"] for r in model_results) / n,
            "mae": sum(r["mae"] for r in model_results) / n,
            "mape": sum(r["mape"] for r in model_results) / n,
            "training_time_seconds": sum(r["training_time_seconds"] for r in model_results) / n,
        }

    for model_type in MODEL_TYPES:
        avg = averages[model_type]
        print(
            f"{model_type.upper():<6} ortalama -> RMSE=${avg['rmse']:.2f}  MAE=${avg['mae']:.2f}  "
            f"MAPE=%{avg['mape']:.2f}  Egitim suresi={avg['training_time_seconds']:.1f}s"
        )

    lower_error_model = min(averages, key=lambda m: averages[m]["mape"])
    faster_model = min(averages, key=lambda m: averages[m]["training_time_seconds"])

    print(
        f"\nOrtalama MAPE'ye gore daha dusuk hata veren model: {lower_error_model.upper()} "
        f"(%{averages[lower_error_model]['mape']:.2f} vs %{averages[[m for m in MODEL_TYPES if m != lower_error_model][0]]['mape']:.2f})"
    )
    print(
        f"Ortalama egitim suresine gore daha hizli egitilen model: {faster_model.upper()} "
        f"({averages[faster_model]['training_time_seconds']:.1f}s vs "
        f"{averages[[m for m in MODEL_TYPES if m != faster_model][0]]['training_time_seconds']:.1f}s)"
    )


def save_summary_csv(results: list[dict]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ticker", "model_type", "rmse", "mae", "mape", "training_time_seconds"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nOzet tablo '{SUMMARY_PATH}' olarak kaydedildi.")


def main() -> None:
    results = []
    for ticker in TICKERS:
        results.extend(run_pipeline_for_ticker(ticker))

    print(f"\n{'=' * 60}\nOzet Sonuclar\n{'=' * 60}")
    print_summary(results)
    save_summary_csv(results)
    print_model_comparison(results)


if __name__ == "__main__":
    main()
