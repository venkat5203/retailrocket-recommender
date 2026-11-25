import numpy as np
from scipy.sparse import csr_matrix

from recsys.models import ItemKNNRecommender, PopularityRecommender, SVDRecommender


def _matrix():
    # user 0 and 1 both like items 0 and 1; user 2 likes items 2 and 3
    dense = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=float)
    return csr_matrix(dense)


def test_popularity_ranks_by_total_weight():
    X = csr_matrix(np.array([[1, 0, 0], [1, 0, 2], [0, 0, 1]], dtype=float))
    model = PopularityRecommender().fit(X)
    scores = model.score(X[:1])
    assert scores.shape == (1, 3)
    np.testing.assert_allclose(scores[0], [2.0, 0.0, 3.0])


def test_item_knn_recommends_co_occurring_item():
    X = _matrix()
    model = ItemKNNRecommender(n_neighbors=2).fit(X)
    user = csr_matrix(np.array([[1, 0, 0, 0]], dtype=float))  # only item 0
    scores = model.score(user)[0]
    assert scores[1] > 0
    assert scores[2] == 0 and scores[3] == 0


def test_item_knn_self_similarity_of_identical_items_is_one():
    X = _matrix()
    model = ItemKNNRecommender(n_neighbors=3).fit(X)
    assert np.isclose(model.similarity_[0, 1], 1.0)


def test_svd_output_shape():
    X = _matrix()
    model = SVDRecommender(n_components=2, random_state=0).fit(X)
    scores = model.score(X)
    assert scores.shape == X.shape
