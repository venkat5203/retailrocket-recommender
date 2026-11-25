"""Loaders for the raw Retailrocket CSV files."""
from __future__ import annotations

import pandas as pd

from . import config


def load_events(path=config.EVENTS_FILE) -> pd.DataFrame:
    """Load user events; timestamps are milliseconds since epoch."""
    df = pd.read_csv(
        path,
        dtype={"visitorid": "int64", "itemid": "int64", "event": "string"},
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def load_category_tree(path=config.CATEGORY_FILE) -> pd.DataFrame:
    """Category hierarchy: categoryid -> parentid (root categories have no parent)."""
    return pd.read_csv(path, dtype={"categoryid": "int64", "parentid": "Int64"})


def load_item_categories(paths=config.ITEM_PROPERTY_FILES, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Most recent categoryid for every item.

    The item property files total ~900 MB, so they are streamed in chunks and only
    rows with property == 'categoryid' are kept.
    """
    parts = []
    for path in paths:
        reader = pd.read_csv(
            path,
            chunksize=chunksize,
            dtype={"itemid": "int64", "property": "string", "value": "string"},
        )
        for chunk in reader:
            sub = chunk.loc[chunk["property"] == "categoryid", ["timestamp", "itemid", "value"]]
            parts.append(sub)

    df = pd.concat(parts, ignore_index=True)
    df["categoryid"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["categoryid"])
    df = df.sort_values("timestamp").drop_duplicates("itemid", keep="last")
    df["categoryid"] = df["categoryid"].astype("int64")
    return df[["itemid", "categoryid"]].reset_index(drop=True)
