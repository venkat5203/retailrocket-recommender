"""Train the baseline models on a temporal split and report ranking metrics."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from recsys import config  # noqa: E402
from recsys.data_loader import load_events  # noqa: E402
from recsys.evaluation import evaluate_model  # noqa: E402
from recsys.models import ItemKNNRecommender, PopularityRecommender, SVDRecommender  # noqa: E402
from recsys.preprocessing import (  # noqa: E402
    aggregate_interactions,
    build_index,
    clean_events,
    k_core_filter,
    temporal_split,
    to_sparse,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=config.MAX_EVAL_USERS)
    args = parser.parse_args()

    events = clean_events(load_events())
    train, test, cutoff = temporal_split(events, config.TEST_FRACTION)
    train = k_core_filter(train, config.MIN_USER_INTERACTIONS, config.MIN_ITEM_INTERACTIONS)
    test = test[test["visitorid"].isin(train["visitorid"]) & test["itemid"].isin(train["itemid"])]

    train_pairs = aggregate_interactions(train)
    users, items = build_index(train_pairs)
    X_train = to_sparse(train_pairs, users, items, "weight")

    test_pos = (
        test[test["event"].isin(config.POSITIVE_EVENTS)][["visitorid", "itemid"]]
        .drop_duplicates()
        .assign(value=1.0)
    )
    T_test = to_sparse(test_pos, users, items, "value")

    print(f"Cutoff timestamp : {cutoff}")
    print(f"Train events     : {len(train):,}  (users={X_train.shape[0]:,}, items={X_train.shape[1]:,})")
    print(f"Test events      : {len(test):,}  (positive pairs={T_test.nnz:,})")

    models = [
        PopularityRecommender(),
        ItemKNNRecommender(n_neighbors=config.KNN_NEIGHBORS),
        SVDRecommender(n_components=config.SVD_COMPONENTS, random_state=config.RANDOM_SEED),
    ]

    rows = []
    for model in models:
        print(f"Evaluating {model.name} ...")
        model.fit(X_train)
        metrics = evaluate_model(
            model, X_train, T_test,
            k_values=config.TOP_K,
            max_users=args.max_users,
            batch_size=config.EVAL_BATCH_SIZE,
            seed=config.RANDOM_SEED,
        )
        rows.append({"model": model.name, **metrics})

    table = pd.DataFrame(rows).set_index("model")
    print("\nResults:")
    print(table.T.round(4).to_string())

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.round(6).to_csv(config.REPORTS_DIR / "metrics.csv")
    with open(config.REPORTS_DIR / "metrics.json", "w") as f:
        json.dump({"cutoff": str(cutoff), "results": table.round(6).to_dict(orient="index")}, f, indent=2)
    print(f"\nSaved to {config.REPORTS_DIR / 'metrics.csv'}")


if __name__ == "__main__":
    main()
