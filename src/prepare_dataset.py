import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLUMNS = ["Close", "MA7", "MA21", "Volume", "daily_return"]
TARGET_COLUMN = "Close"
SEQUENCE_LENGTH = 30
TEST_RATIO = 0.15


def load_data(ticker: str) -> pd.DataFrame:
    input_path = DATA_DIR / f"{ticker}_processed.csv"
    df = pd.read_csv(input_path, index_col="Date", parse_dates=True)
    df = df.sort_index()
    before = len(df)
    df = df.dropna()
    print(f"[{ticker}] Rolling window NaN satirlari dusuruldu: {before - len(df)} satir ({before} -> {len(df)})")
    return df


def split_train_test(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - TEST_RATIO))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(
        f"[{ticker}] Train/test split (shuffle yok, son %{TEST_RATIO * 100:.0f} test): "
        f"train={len(train_df)}, test={len(test_df)}"
    )
    return train_df, test_df


def scale_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    # Scaler sadece train verisiyle fit edilir; test setinin istatistiklerinin
    # train'e sizmasini (data leakage) onlemek icin.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[FEATURE_COLUMNS])

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[FEATURE_COLUMNS] = scaler.transform(train_df[FEATURE_COLUMNS])
    test_scaled[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    return train_scaled, test_scaled, scaler


def create_sequences(df: pd.DataFrame, seq_length: int = SEQUENCE_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    feature_values = df[FEATURE_COLUMNS].values
    target_values = df[TARGET_COLUMN].values

    X, y = [], []
    for i in range(len(df) - seq_length):
        X.append(feature_values[i : i + seq_length])
        y.append(target_values[i + seq_length])

    return np.array(X), np.array(y)


def prepare_dataset(ticker: str) -> None:
    df = load_data(ticker)
    train_df, test_df = split_train_test(df, ticker)
    train_scaled, test_scaled, scaler = scale_features(train_df, test_df)

    X_train, y_train = create_sequences(train_scaled)
    X_test, y_test = create_sequences(test_scaled)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / f"{ticker}_X_train.npy", X_train)
    np.save(DATA_DIR / f"{ticker}_y_train.npy", y_train)
    np.save(DATA_DIR / f"{ticker}_X_test.npy", X_test)
    np.save(DATA_DIR / f"{ticker}_y_test.npy", y_test)

    with open(DATA_DIR / f"{ticker}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"[{ticker}] X_train {X_train.shape}, X_test {X_test.shape} -> data/ klasorune kaydedildi.\n")


if __name__ == "__main__":
    prepare_dataset("AAPL")
