import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def _sq_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return cdist(X, centers, metric="sqeuclidean")

def enforce_min_dwell(seq: np.ndarray, min_dwell: int) -> np.ndarray:
    out = seq.copy()
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(out):
            cur = out[i]
            j = i
            while j < len(out) and out[j] == cur:
                j += 1
            if (j - i) < min_dwell:
                replacement = out[j] if i == 0 else out[i - 1]
                out[i:j] = replacement
                changed = True
            i = j
    return out

def viterbi_global(X: np.ndarray, centers: np.ndarray, jump_penalty: float) -> np.ndarray:
    n, K = len(X), len(centers)
    D = _sq_distances(X, centers)

    cost = np.full((n, K), np.inf)
    prev = np.full((n, K), -1, dtype=int)
    cost[0] = D[0]

    for t in range(1, n):
        for s in range(K):
            stay = cost[t - 1, s] + D[t, s]
            other_costs = [cost[t - 1, s2] for s2 in range(K) if s2 != s]
            best_switch = min(other_costs) + jump_penalty + D[t, s]
            if stay <= best_switch:
                cost[t, s], prev[t, s] = stay, s
            else:
                src = int(np.argmin([cost[t - 1, s2] if s2 != s else np.inf for s2 in range(K)]))
                src = src if src < s else src + 1
                cost[t, s], prev[t, s] = best_switch, src

    labels = np.empty(n, dtype=int)
    labels[-1] = int(np.argmin(cost[-1]))
    for t in range(n - 2, -1, -1):
        labels[t] = prev[t + 1, labels[t + 1]]
    return labels

def viterbi_causal(X: np.ndarray, centers: np.ndarray, jump_penalty: float) -> np.ndarray:
    n, K = len(X), len(centers)
    D = _sq_distances(X, centers)
    running = D[0].copy()
    labels = np.empty(n, dtype=int)
    labels[0] = int(np.argmin(running))

    for t in range(1, n):
        new_cost = np.empty(K)
        for s in range(K):
            stay = running[s] + D[t, s]
            best_switch = min(
                running[s2] + jump_penalty + D[t, s]
                for s2 in range(K) if s2 != s
            )
            new_cost[s] = min(stay, best_switch)
        running = new_cost
        labels[t] = int(np.argmin(running))

    return labels

def fit_jump_model(
    X: np.ndarray,
    n_states: int,
    jump_penalty: float,
    min_dwell: int = 1,
    n_init: int = 10,
    max_iter: int = 500,
    tol: float = 1e-6,
    global_viterbi: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    decode = viterbi_global if global_viterbi else viterbi_causal
    n, _ = X.shape
    best_loss, best_centers, best_labels = np.inf, None, None

    for _ in range(n_init):
        centers = X[np.random.choice(n, n_states, replace=False)].copy()
        for _ in range(max_iter):
            labels = decode(X, centers, jump_penalty)
            if min_dwell > 1:
                labels = enforce_min_dwell(labels, min_dwell)
            new_centers = np.array([
                X[labels == s].mean(axis=0) if (labels == s).any() else centers[s]
                for s in range(n_states)
            ])
            if np.linalg.norm(new_centers - centers) < tol:
                centers = new_centers
                break
            centers = new_centers

        D = _sq_distances(X, centers)
        transitions = int((labels[:-1] != labels[1:]).sum())
        loss = D[np.arange(n), labels].sum() + jump_penalty * transitions
        if loss < best_loss:
            best_loss = loss
            best_centers = centers.copy()
            best_labels = labels.copy()
    return best_centers, best_labels

def align_labels_to_reference(
    ref_centers: np.ndarray,
    other_centers: np.ndarray,
    other_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = len(ref_centers)
    cost = cdist(ref_centers, other_centers, metric="sqeuclidean")
    ref_idx, other_idx = linear_sum_assignment(cost)
    mapping = np.zeros(K, dtype=int)
    for r, o in zip(ref_idx, other_idx):
        mapping[o] = r
    aligned_labels = mapping[other_labels]
    aligned_centers = np.zeros_like(other_centers)
    for o, r in enumerate(mapping):
        aligned_centers[r] = other_centers[o]

    return aligned_labels, aligned_centers, mapping
