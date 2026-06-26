import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from config import N_REGIMES, JUMP_PENALTY, MIN_DWELL, RESULTS_DIR, RAW_DATA_PATH
from utils import load_and_resample, load_model, save_model, regime_transition_stats
from features import compute_log_returns_and_flow, extract_features
from jump_model import fit_jump_model, viterbi_causal, align_labels_to_reference

def macro_roc_auc(y_true, y_pred, n_classes):
    y_true_b = label_binarize(y_true, classes=list(range(n_classes)))
    y_pred_b = label_binarize(y_pred, classes=list(range(n_classes)))
    present = [c for c in range(n_classes) if y_true_b[:, c].std() > 0]
    if not present:
        return float("nan")
    return roc_auc_score(y_true_b[:, present], y_pred_b[:, present],
                         average="macro", multi_class="ovr")

def train_and_eval(train_X, test_X, n_regimes, jump_penalty, n_init=5):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_X)
    test_scaled = scaler.transform(test_X)
    centers, train_labels = fit_jump_model(
        train_scaled, n_regimes, jump_penalty,
        n_init=n_init, global_viterbi=False,
    )
    test_labels = viterbi_causal(test_scaled, centers, jump_penalty)
    return centers, scaler, train_labels, test_labels

def main():
    df = load_and_resample(RAW_DATA_PATH)
    df = compute_log_returns_and_flow(df)
    features, timestamps = extract_features(df)
    baseline = load_model(os.path.join(RESULTS_DIR, "btc_jump_model_baseline.pkl"))
    ref_centers_unscaled = baseline["centers_unscaled"]
    oracle_labels = baseline["labels"]
    split = len(features) // 2
    centers1, scaler1, _, test1 = train_and_eval(
        features[:split], features[split:], N_REGIMES, JUMP_PENALTY
    )
    centers1_unscaled = scaler1.inverse_transform(centers1)
    test1_aligned, _, mapping1 = align_labels_to_reference(ref_centers_unscaled, centers1_unscaled, test1)
    oracle_second = oracle_labels[split: split + len(test1_aligned)]
    auc1 = macro_roc_auc(oracle_second, test1_aligned, N_REGIMES)
    centers2, scaler2, _, test2 = train_and_eval(
        features[split:], features[:split], N_REGIMES, JUMP_PENALTY
    )
    centers2_unscaled = scaler2.inverse_transform(centers2)
    test2_aligned, _, mapping2 = align_labels_to_reference(ref_centers_unscaled, centers2_unscaled, test2)
    oracle_first = oracle_labels[:len(test2_aligned)]
    auc2 = macro_roc_auc(oracle_first, test2_aligned, N_REGIMES)
    print(f"Model 1 (train: first half → test: second half)  ROC-AUC = {auc1:.4f}")
    print(f"Model 2 (train: second half → test: first half)  ROC-AUC = {auc2:.4f}")
    print(f"Label mapping Model 1: {dict(enumerate(mapping1))}")
    print(f"Label mapping Model 2: {dict(enumerate(mapping2))}")
    for name, payload in [
        ("model1", {"centers": centers1, "scaler": scaler1, "labels_test": test1_aligned, "auc": auc1, "mapping": mapping1}),
        ("model2", {"centers": centers2, "scaler": scaler2, "labels_test": test2_aligned, "auc": auc2, "mapping": mapping2}),
    ]:
        save_model(payload, os.path.join(RESULTS_DIR, f"btc_jump_{name}.pkl"))

if __name__ == "__main__":
    main()
