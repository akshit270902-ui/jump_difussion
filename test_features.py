import numpy as np
import pandas as pd
import pytest
from src.features import extract_features, compute_log_returns_and_flow, _signature_level2

def _make_dummy_df(n=200):
    np.random.seed(42)
    close = 30000 * np.exp(np.random.randn(n).cumsum() * 0.002)
    volume = np.abs(np.random.randn(n)) * 1000 + 500
    taker_buy = volume * np.random.uniform(0.3, 0.7, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="1H"),
        "close": close,
        "volume": volume,
        "taker_buy_volume": taker_buy,
        "log_ret": np.log(close / np.roll(close, 1)),
        "vol_delta": 2 * taker_buy - volume,
    }).iloc[1:].reset_index(drop=True)

def test_signature_shape():
    path = np.random.randn(20, 4)
    sig = _signature_level2(path)
    assert sig.shape == (20,)

def test_signature_single_step():
    path = np.random.randn(2, 4)
    sig = _signature_level2(path)
    assert sig.shape == (20,)
    assert not np.any(np.isnan(sig))

def test_feature_output_shape():
    df = _make_dummy_df(200)
    feats, ts = extract_features(df)
    max_w = 100
    assert feats.shape[0] == len(df) - max_w
    assert len(ts) == feats.shape[0]

def test_no_nans_in_features():
    df = _make_dummy_df(200)
    feats, _ = extract_features(df)
    assert not np.any(np.isnan(feats))

def test_feature_determinism():
    df = _make_dummy_df(150)
    f1, _ = extract_features(df)
    f2, _ = extract_features(df)
    np.testing.assert_array_equal(f1, f2)
