import os
import pickle
import pandas as pd
import numpy as np
from config import RESAMPLE_FREQ, CUTOFF_DATE

OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "taker_buy_volume": "sum",
}
def load_and_resample(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df.columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
    ][:len(df.columns)]
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    cutoff = pd.to_datetime(CUTOFF_DATE, utc=True)
    df = df[df.index >= cutoff]
    df_h = df.resample(RESAMPLE_FREQ).agg(OHLCV_AGG).dropna()
    df_h = df_h.reset_index()
    df_h["timestamp"] = pd.to_datetime(df_h["timestamp"])
    return df_h

def save_model(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved → {path}")

def load_model(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

def regime_transition_stats(labels: np.ndarray) -> dict:
    transitions = int((labels[:-1] != labels[1:]).sum())
    unique, counts = np.unique(labels, return_counts=True)
    return {
        "n_transitions": transitions,
        "transition_rate": transitions / max(len(labels) - 1, 1),
        "regime_counts": dict(zip(unique.tolist(), counts.tolist())),
        "regime_fractions": dict(zip(unique.tolist(), (counts / counts.sum()).round(4).tolist())),
    }
