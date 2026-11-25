"""Ranking metrics for top-K recommendation on a temporal test split."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def _user_metrics(hits: np.ndarray, n_relevant: int, k_values) -> dict:
    """Metrics for one user. `hits` is a 0/1 array over the ranked recommendations."""
    out = {}
    for k in k_values:
        h = hits[:k]
        n_hit = h.sum()
        out[f"precision@{k}"] = n_hit / k
        out[f"recall@{k}"] = n_hit / n_relevant
        out[f"hit_rate@{k}"] = float(n_hit > 0)

        discount = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = (h * discount).sum()
        idcg = discount[: min(n_relevant, k)].sum()
        out[f"ndcg@{k}"] = dcg / idcg

        precision_at_i = np.cumsum(h) / np.arange(1, k + 1)
        out[f"map@{k}"] = (precision_at_i * h).sum() / min(n_relevant, k)
    return out


def evaluate_model(
    model,
    X_train: csr_matrix,
    T_test: csr_matrix,
    k_values=(5, 10, 20),
    max_users: int = 5000,
    batch_size: int = 500,
    seed: int = 42,
) -> dict:
    """Average ranking metrics over users who have both training history and test positives.

    Items the user already interacted with in training are excluded from recommendations.
    """
    rel_counts = np.diff(T_test.indptr)
    train_counts = np.diff(X_train.indptr)
    eligible = np.flatnonzero((rel_counts > 0) & (train_counts > 0))
    if len(eligible) > max_users:
        eligible = np.random.default_rng(seed).choice(eligible, size=max_users, replace=False)

    K = max(k_values)
    n_items = X_train.shape[1]
    recommended = np.zeros(n_items, dtype=bool)
    totals: dict[str, float] = {}

    for start in range(0, len(eligible), batch_size):
        batch = eligible[start : start + batch_size]
        X_batch = X_train[batch]
        scores = np.asarray(model.score(X_batch), dtype=float)
        scores[X_batch.toarray() > 0] = -np.inf  # never recommend already-seen items

        part = np.argpartition(-scores, K - 1, axis=1)[:, :K]
        part_scores = np.take_along_axis(scores, part, axis=1)
        top = np.take_along_axis(part, np.argsort(-part_scores, axis=1), axis=1)
        recommended[top.ravel()] = True

        for row, user in enumerate(batch):
            rel = T_test.indices[T_test.indptr[user] : T_test.indptr[user + 1]]
            hits = np.isin(top[row], rel).astype(float)
            for name, value in _user_metrics(hits, len(rel), k_values).items():
                totals[name] = totals.get(name, 0.0) + value

    n = len(eligible)
    results = {name: total / n for name, total in totals.items()}
    results["coverage"] = recommended.mean()
    results["n_eval_users"] = n
    return results
