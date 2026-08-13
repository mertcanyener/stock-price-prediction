from pathlib import Path

import pandas as pd
import yfinance as yf

TICKER = "AAPL"
PERIOD = "5y"
INTERVAL = "1d"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "aapl_raw.csv"


def fetch_data() -> None:
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH)

    print(f"{len(df)} satir veri indirildi ve '{OUTPUT_PATH}' dosyasina kaydedildi.\n")
    print("Ilk 5 satir:")
    print(df.head())
    print("\nSon 5 satir:")
    print(df.tail())


if __name__ == "__main__":
    fetch_data()
