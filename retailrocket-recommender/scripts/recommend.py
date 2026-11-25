"""Train on all available history and print top-N recommendations for one visitor."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from recsys import config  # noqa: E402
from recsys.data_loader import load_events, load_item_categories  # noqa: E402
from recsys.models import ItemKNNRecommender, PopularityRecommender, SVDRecommender  # noqa: E402
from recsys.preprocessing import (  # noqa: E402
    aggregate_interactions,
    build_index,
    clean_events,
    k_core_filter,
    to_sparse,
)

MODELS = {
    "item_knn": lambda: ItemKNNRecommender(n_neighbors=config.KNN_NEIGHBORS),
    "svd": lambda: SVDRecommender(n_components=config.SVD_COMPONENTS, random_state=config.RANDOM_SEED),
    "popularity": PopularityRecommender,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visitor", type=int, required=True, help="visitorid to recommend for")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--model", choices=MODELS, default="item_knn")
    args = parser.parse_args()

    events = clean_events(load_events())
    events = k_core_filter(events, config.MIN_USER_INTERACTIONS, config.MIN_ITEM_INTERACTIONS)
    pairs = aggregate_interactions(events)
    users, items = build_index(pairs)
    X = to_sparse(pairs, users, items, "weight")

    if args.visitor not in users:
        sys.exit(f"Visitor {args.visitor} has no usable history after filtering.")

    model = MODELS[args.model]().fit(X)
    row = users.get_loc(args.visitor)
    X_user = X[row]
    scores = np.asarray(model.score(X_user), dtype=float).ravel()
    scores[X_user.toarray().ravel() > 0] = -np.inf  # hide items already seen

    top_idx = np.argsort(-scores)[: args.top_n]
    result = pd.DataFrame({"itemid": items[top_idx], "score": scores[top_idx].round(4)})
    cats = load_item_categories().set_index("itemid")["categoryid"]
    result["categoryid"] = result["itemid"].map(cats)

    print(f"Top {args.top_n} recommendations for visitor {args.visitor} ({args.model}):")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
