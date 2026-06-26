import numpy as np
import pandas as pd
from config import WINDOWS

def compute_log_returns_and_flow(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["vol_delta"] = 2 * df["taker_buy_volume"] - df["volume"]
    return df.dropna().reset_index(drop=True)

def _signature_level2(path: np.ndarray) -> np.ndarray:
    dX = np.diff(path, axis=0)
    level1 = path[-1] - path[0]

    if len(dX) > 1:
        C = np.cumsum(dX, axis=0)          
        level2 = np.dot(C[:-1].T, dX[1:]).flatten()
    else:
        level2 = np.zeros(path.shape[1] ** 2)

    return np.concatenate([level1, level2])

def _build_path(slice_df: pd.DataFrame, w: int) -> np.ndarray:
    p = slice_df["log_ret"].cumsum().values
    v = slice_df["volume"].cumsum().values
    vd = slice_df["vol_delta"].cumsum().values
    t = np.arange(w, dtype=float)
    path = np.column_stack([t, p, v, vd])
    p_min = path.min(axis=0)
    p_max = path.max(axis=0)
    denom = p_max - p_min
    denom[denom == 0] = 1.0
    return (path - p_min) / denom

def extract_features(df: pd.DataFrame, windows: list[int] = WINDOWS) -> tuple[np.ndarray, pd.Series]:
    max_w = max(windows)
    feature_rows = []

    for i in range(max_w, len(df)):
        row = []
        for w in windows:
            slice_df = df.iloc[i - w + 1: i + 1]
            path = _build_path(slice_df, w)
            row.extend(_signature_level2(path))
        feature_rows.append(row)

    features = np.array(feature_rows)
    timestamps = df["timestamp"].iloc[max_w:].reset_index(drop=True)
    return features, timestamps
