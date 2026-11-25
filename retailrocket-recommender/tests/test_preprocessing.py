import pandas as pd

from recsys.preprocessing import build_index, clean_events, k_core_filter, temporal_split, to_sparse


def _events():
    return pd.DataFrame({
        "timestamp": pd.to_datetime([1, 2, 3, 4, 5, 6], unit="s"),
        "visitorid": [1, 1, 2, 2, 3, 3],
        "itemid": [10, 11, 10, 11, 10, 99],
        "event": ["view", "addtocart", "view", "transaction", "view", "bogus"],
    })


def test_clean_events_drops_unknown_events_and_adds_weights():
    df = clean_events(_events())
    assert "bogus" not in set(df["event"])
    assert df.loc[df["event"] == "transaction", "weight"].iloc[0] == 5.0


def test_temporal_split_is_disjoint_and_ordered():
    df = clean_events(_events())
    train, test, cutoff = temporal_split(df, test_fraction=0.5)
    assert train["timestamp"].max() < cutoff <= test["timestamp"].min()
    assert len(train) + len(test) == len(df)


def test_k_core_filter_removes_rare_items_and_users():
    df = pd.DataFrame({
        "visitorid": [1, 1, 2, 2, 3],
        "itemid": [10, 11, 10, 11, 10],
        "event": ["view"] * 5,
        "weight": [1.0] * 5,
    })
    out = k_core_filter(df, min_user=2, min_item=2)
    assert set(out["itemid"]) == {10, 11}
    out = k_core_filter(df, min_user=2, min_item=3)
    assert out.empty


def test_to_sparse_shape_and_values():
    pairs = pd.DataFrame({"visitorid": [1, 2], "itemid": [10, 11], "weight": [3.0, 5.0]})
    users, items = build_index(pairs)
    X = to_sparse(pairs, users, items)
    assert X.shape == (2, 2)
    assert X.sum() == 8.0
