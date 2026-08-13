from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_PATH = DATA_DIR / "aapl_raw.csv"
OUTPUT_PATH = DATA_DIR / "aapl_processed.csv"

MA_SHORT_WINDOW = 7
MA_LONG_WINDOW = 21
OUTLIER_Z_THRESHOLD = 3


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_counts = df.isna().sum()
    total_missing = missing_counts.sum()

    print("Eksik deger kontrolu:")
    if total_missing == 0:
        print("  Eksik deger bulunamadi.\n")
        return df

    print(missing_counts[missing_counts > 0].to_string())
    df = df.ffill().bfill()
    print(f"  Toplam {total_missing} eksik deger, forward-fill/backward-fill ile dolduruldu.\n")
    return df


def check_outliers(df: pd.DataFrame) -> None:
    returns = df["Close"].pct_change()
    z_scores = (returns - returns.mean()) / returns.std()
    outliers = df.loc[z_scores.abs() > OUTLIER_Z_THRESHOLD, ["Close"]].copy()
    outliers["daily_return"] = returns[z_scores.abs() > OUTLIER_Z_THRESHOLD]

    print(f"Outlier kontrolu (|z-score| > {OUTLIER_Z_THRESHOLD} gunluk getiri):")
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


def preprocess() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, index_col="Date", parse_dates=True)
    df = df.sort_index()

    df = check_missing_values(df)
    check_outliers(df)
    df = add_features(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH)

    print(f"{len(df)} satir islendi ve '{OUTPUT_PATH}' dosyasina kaydedildi.\n")
    print(f"Kolonlar: {list(df.columns)}\n")
    print("Ilk 5 satir:")
    print(df.head())
    print("\nSon 5 satir:")
    print(df.tail())

    return df


if __name__ == "__main__":
    preprocess()
