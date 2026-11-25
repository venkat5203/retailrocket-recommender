import numpy as np
from scipy.sparse import csr_matrix

from recsys.evaluation import _user_metrics, evaluate_model


class OracleModel:
    """Scores the held-out positives highest, so metrics should be perfect."""

    def __init__(self, T):
        self.T = T.toarray()

    def score(self, X_users):
        return self.T[: X_users.shape[0]].astype(float)


def test_user_metrics_perfect_ranking():
    hits = np.array([1.0, 1.0, 0.0, 0.0])
    m = _user_metrics(hits, n_relevant=2, k_values=(2, 4))
    assert m["precision@2"] == 1.0
    assert m["recall@2"] == 1.0
    assert m["ndcg@2"] == 1.0
    assert m["map@2"] == 1.0


def test_user_metrics_miss():
    hits = np.zeros(3)
    m = _user_metrics(hits, n_relevant=1, k_values=(3,))
    assert m["hit_rate@3"] == 0.0
    assert m["ndcg@3"] == 0.0


def test_evaluate_model_with_oracle_is_perfect_and_masks_seen_items():
    X_train = csr_matrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float))
    T_test = csr_matrix(np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=float))
    model = OracleModel(T_test)
    res = evaluate_model(model, X_train, T_test, k_values=(1,), max_users=10, batch_size=10)
    assert res["n_eval_users"] == 2
    assert res["hit_rate@1"] == 1.0
    assert res["ndcg@1"] == 1.0
