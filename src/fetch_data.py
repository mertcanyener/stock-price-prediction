from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "NVDA", "META", "PLTR", "GOOGL", "TTWO", "TSLA", "AMZN", "AMD"]
PERIOD = "5y"
INTERVAL = "1d"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"{ticker}_raw.csv"
    df.to_csv(output_path)

    print(f"[{ticker}] {len(df)} satir veri indirildi ve '{output_path}' dosyasina kaydedildi.")
    return df


def fetch_all(tickers: list[str] = TICKERS) -> None:
    for ticker in tickers:
        fetch_data(ticker)


if __name__ == "__main__":
    fetch_all()
