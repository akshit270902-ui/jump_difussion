import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
import pandas as pd
from utils import load_model
from features import compute_log_returns_and_flow, extract_features
from jump_model import viterbi_causal
from config import WINDOWS

def predict(model_path: str, data_path: str) -> pd.DataFrame:
    saved = load_model(model_path)
    centers = saved["centers"]
    scaler = saved["scaler"]
    jump_penalty = saved.get("jump_penalty", 40.0)
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df[df.columns[0]])
    df = compute_log_returns_and_flow(df)
    features, timestamps = extract_features(df, windows=WINDOWS)
    X_scaled = scaler.transform(features)
    labels = viterbi_causal(X_scaled, centers, jump_penalty)
    result = df.iloc[max(WINDOWS):].copy().reset_index(drop=True)
    result["regime"] = labels
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="results/predictions.csv")
    args = parser.parse_args()
    out = predict(args.model, args.data)
    out.to_csv(args.out, index=False)
    print(f"Predictions saved to {args.out}")
    print(out[["timestamp", "close", "regime"]].tail(10))
