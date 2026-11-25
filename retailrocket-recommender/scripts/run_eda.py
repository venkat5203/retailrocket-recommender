"""Exploratory data analysis: summary stats, funnel, daily activity and top categories."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from recsys import config  # noqa: E402
from recsys.data_loader import load_events, load_item_categories  # noqa: E402


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events()

    viewers = set(events.loc[events["event"] == "view", "visitorid"])
    carters = set(events.loc[events["event"] == "addtocart", "visitorid"])
    buyers = set(events.loc[events["event"] == "transaction", "visitorid"])

    summary = {
        "n_events": int(len(events)),
        "n_visitors": int(events["visitorid"].nunique()),
        "n_items": int(events["itemid"].nunique()),
        "date_range": [str(events["timestamp"].min()), str(events["timestamp"].max())],
        "event_counts": {k: int(v) for k, v in events["event"].value_counts().items()},
        "funnel": {
            "visitors_viewed": len(viewers),
            "visitors_added_to_cart": len(carters),
            "visitors_purchased": len(buyers),
            "view_to_cart_rate": round(len(carters & viewers) / len(viewers), 4),
            "cart_to_purchase_rate": round(len(buyers & carters) / len(carters), 4),
        },
        "events_per_visitor_median": float(events.groupby("visitorid").size().median()),
    }

    daily = events.groupby([pd.Grouper(key="timestamp", freq="D"), "event"]).size().unstack(fill_value=0)
    ax = daily.plot(figsize=(11, 4), title="Daily events by type")
    ax.set_ylabel("events")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "daily_events.png", dpi=120)
    plt.close()

    funnel_vals = [len(viewers), len(carters), len(buyers)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Viewed", "Added to cart", "Purchased"], funnel_vals, color=["#8ea9c1", "#e0a458", "#5c8d5a"])
    ax.set_title("Visitor funnel")
    ax.set_ylabel("unique visitors")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "funnel.png", dpi=120)
    plt.close()

    print("Loading item categories (streams ~900 MB of item properties)...")
    item_cats = load_item_categories()
    views = events.loc[events["event"] == "view", ["itemid"]].merge(item_cats, on="itemid", how="left")
    top_cats = views["categoryid"].dropna().astype("int64").value_counts().head(10)
    summary["top_categories_by_views"] = {str(k): int(v) for k, v in top_cats.items()}
    summary["share_of_views_with_category"] = round(views["categoryid"].notna().mean(), 4)

    fig, ax = plt.subplots(figsize=(8, 4))
    top_cats.iloc[::-1].plot(kind="barh", ax=ax, color="#8ea9c1")
    ax.set_title("Top 10 categories by view events")
    ax.set_xlabel("view events")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "top_categories.png", dpi=120)
    plt.close()

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
