"""Cleaning, splitting and conversion of events into sparse user-item matrices."""
from __future__ import annotations

import pandas as pd
from scipy.sparse import csr_matrix

from . import config


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate events, keep known event types and add an implicit weight."""
    df = events.drop_duplicates(subset=["visitorid", "itemid", "event", "timestamp"])
    df = df[df["event"].isin(config.EVENT_WEIGHTS)].copy()
    df["weight"] = df["event"].map(config.EVENT_WEIGHTS).astype("float64")
    return df.reset_index(drop=True)


def temporal_split(events: pd.DataFrame, test_fraction: float):
    """Split by time: the most recent `test_fraction` of events becomes the test set."""
    ordered = events["timestamp"].sort_values()
    cutoff = ordered.iloc[int(len(ordered) * (1 - test_fraction))]
    train = events[events["timestamp"] < cutoff]
    test = events[events["timestamp"] >= cutoff]
    return train, test, cutoff


def k_core_filter(events: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Repeatedly drop users and items with too few interactions until stable."""
    while True:
        n_before = len(events)
        events = events[events.groupby("itemid")["visitorid"].transform("size") >= min_item]
        events = events[events.groupby("visitorid")["itemid"].transform("size") >= min_user]
        if len(events) == n_before:
            return events


def aggregate_interactions(events: pd.DataFrame, value_col: str = "weight") -> pd.DataFrame:
    """One row per (visitor, item) using the strongest event weight observed."""
    return events.groupby(["visitorid", "itemid"], as_index=False)[value_col].max()


def build_index(pairs: pd.DataFrame):
    """Index objects mapping raw visitor and item ids to matrix rows and columns."""
    users = pd.Index(pairs["visitorid"].unique(), name="visitorid")
    items = pd.Index(pairs["itemid"].unique(), name="itemid")
    return users, items


def to_sparse(pairs: pd.DataFrame, users: pd.Index, items: pd.Index, value_col: str = "weight") -> csr_matrix:
    """Build a (users x items) CSR matrix. Pairs with unknown ids are ignored."""
    u = users.get_indexer(pairs["visitorid"])
    i = items.get_indexer(pairs["itemid"])
    keep = (u >= 0) & (i >= 0)
    values = pairs[value_col].to_numpy()[keep]
    return csr_matrix((values, (u[keep], i[keep])), shape=(len(users), len(items)))
