"""Recommendation models. Every model exposes fit(X) and score(X_users)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix, diags, vstack
from sklearn.decomposition import TruncatedSVD


class Recommender(ABC):
    name = "base"

    @abstractmethod
    def fit(self, X: csr_matrix) -> "Recommender":
        """Learn from the (users x items) training matrix."""

    @abstractmethod
    def score(self, X_users: csr_matrix) -> np.ndarray:
        """Return a dense (n_users, n_items) array of relevance scores."""


class PopularityRecommender(Recommender):
    """Recommends the same items to everyone, ranked by total interaction weight."""

    name = "popularity"

    def fit(self, X: csr_matrix) -> "PopularityRecommender":
        self.popularity_ = np.asarray(X.sum(axis=0)).ravel()
        return self

    def score(self, X_users: csr_matrix) -> np.ndarray:
        return np.tile(self.popularity_, (X_users.shape[0], 1))


def _keep_top_k(sim: csr_matrix, k: int) -> csr_matrix:
    """Keep only the k largest values in each row of a sparse matrix."""
    sim = sim.tocsr()
    rows, cols, vals = [], [], []
    for r in range(sim.shape[0]):
        start, end = sim.indptr[r], sim.indptr[r + 1]
        idx = sim.indices[start:end]
        v = sim.data[start:end]
        if len(v) > k:
            top = np.argpartition(-v, k)[:k]
            idx, v = idx[top], v[top]
        rows.append(np.full(len(idx), r))
        cols.append(idx)
        vals.append(v)
    return csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=sim.shape,
    )


class ItemKNNRecommender(Recommender):
    """Item-item collaborative filtering with cosine similarity.

    A user's score for item j is the sum of similarities between j and the items
    the user has already interacted with. Similarities are computed in blocks of
    items to keep memory bounded, and only the top-k neighbours per item are kept.
    """

    name = "item_knn"

    def __init__(self, n_neighbors: int = 50, block_size: int = 1000):
        self.n_neighbors = n_neighbors
        self.block_size = block_size

    def fit(self, X: csr_matrix) -> "ItemKNNRecommender":
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        X_norm = (X @ diags(1.0 / norms)).tocsc()
        item_user = X_norm.T.tocsr()

        blocks = []
        n_items = X.shape[1]
        for start in range(0, n_items, self.block_size):
            end = min(start + self.block_size, n_items)
            sim = (item_user[start:end] @ X_norm).tocsr()
            blocks.append(_keep_top_k(sim, self.n_neighbors))
        self.similarity_ = vstack(blocks).tocsr()
        return self

    def score(self, X_users: csr_matrix) -> np.ndarray:
        return (X_users @ self.similarity_).toarray()


class SVDRecommender(Recommender):
    """Latent-factor model: truncated SVD on log-scaled interaction weights."""

    name = "svd"

    def __init__(self, n_components: int = 64, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: csr_matrix) -> "SVDRecommender":
        X_log = X.copy()
        X_log.data = np.log1p(X_log.data)
        k = min(self.n_components, X.shape[1] - 1)
        svd = TruncatedSVD(n_components=k, random_state=self.random_state).fit(X_log)
        self.components_ = svd.components_  # (k, n_items)
        return self

    def score(self, X_users: csr_matrix) -> np.ndarray:
        X_log = X_users.copy()
        X_log.data = np.log1p(X_log.data)
        latent = X_log @ self.components_.T  # (n_users, k)
        return np.asarray(latent @ self.components_)
