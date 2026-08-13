from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MA_SHORT_WINDOW = 7
MA_LONG_WINDOW = 21
OUTLIER_Z_THRESHOLD = 3


def check_missing_values(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    missing_counts = df.isna().sum()
    total_missing = missing_counts.sum()

    print(f"[{ticker}] Eksik deger kontrolu:")
    if total_missing == 0:
        print("  Eksik deger bulunamadi.\n")
        return df

    print(missing_counts[missing_counts > 0].to_string())
    df = df.ffill().bfill()
    print(f"  Toplam {total_missing} eksik deger, forward-fill/backward-fill ile dolduruldu.\n")
    return df


def check_outliers(df: pd.DataFrame, ticker: str) -> None:
    returns = df["Close"].pct_change()
    z_scores = (returns - returns.mean()) / returns.std()
    outliers = df.loc[z_scores.abs() > OUTLIER_Z_THRESHOLD, ["Close"]].copy()
    outliers["daily_return"] = returns[z_scores.abs() > OUTLIER_Z_THRESHOLD]

    print(f"[{ticker}] Outlier kontrolu (|z-score| > {OUTLIER_Z_THRESHOLD} gunluk getiri):")
    if outliers.empty:
        print("  Anormal fiyat sicramasi bulunamadi.\n")
        return

    print(f"  {len(outliers)} adet anormal gunluk getiri tespit edildi (kaldirilmadi, bilgi amacli):")
    print(outliers.to_string())
    print()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["daily_return"] = df["Close"].pct_change()
    df["MA7"] = df["Close"].rolling(window=MA_SHORT_WINDOW).mean()
    df["MA21"] = df["Close"].rolling(window=MA_LONG_WINDOW).mean()
    return df


def preprocess(ticker: str) -> pd.DataFrame:
    input_path = DATA_DIR / f"{ticker}_raw.csv"
    output_path = DATA_DIR / f"{ticker}_processed.csv"

    df = pd.read_csv(input_path, index_col="Date", parse_dates=True)
    df = df.sort_index()

    df = check_missing_values(df, ticker)
    check_outliers(df, ticker)
    df = add_features(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)

    print(f"[{ticker}] {len(df)} satir islendi ve '{output_path}' dosyasina kaydedildi.\n")

    return df


if __name__ == "__main__":
    preprocess("AAPL")
