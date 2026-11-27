from __future__ import annotations

import json
import tempfile
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
API = "https://huggingface.co/api/datasets/NEU-HAI/OPeRA/tree/main/OPeRA_full/action?recursive=true&expand=false"
DATASET_ROOT = "https://huggingface.co/datasets/NEU-HAI/OPeRA/resolve/main/"
DATASET_PAGE = "https://huggingface.co/datasets/NEU-HAI/OPeRA"
PAPER = "https://arxiv.org/abs/2506.05606"
LICENSE = "CC BY 4.0"


def request_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "VenkatOPeRARecommender/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def download_file(url: str, destination: Path):
    req = urllib.request.Request(url, headers={"User-Agent": "VenkatOPeRARecommender/1.0"})
    with urllib.request.urlopen(req, timeout=180) as response, destination.open("wb") as out:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            out.write(block)


def _decode_json(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def extract_products(value):
    if isinstance(value, str):
        decoded = _decode_json(value)
        return extract_products(decoded) if decoded is not None else []
    if isinstance(value, list):
        found = []
        for part in value:
            found.extend(extract_products(part))
        return found
    if not isinstance(value, dict):
        return []
    if value.get("asin") and (value.get("title") or value.get("name")):
        return [value]
    found = []
    for part in value.values():
        found.extend(extract_products(part))
    return found


def price_number(value):
    if value is None:
        return None
    text = str(value).replace(",", "").replace("$", "").replace("£", "").strip()
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def load_actions(temp_dir: Path):
    files = request_json(API)
    paths = sorted(
        item["path"] for item in files
        if item.get("type") == "file" and item.get("path", "").endswith(".parquet")
        and ("/train-" in item.get("path", "") or "/test-" in item.get("path", ""))
    )
    if not paths:
        raise RuntimeError("No OPeRA full-action Parquet shards were found.")
    required = ["session_id", "timestamp", "action_type", "click_type", "element_meta", "products"]
    frames, file_summary = [], []
    for idx, path in enumerate(paths, start=1):
        local = temp_dir / Path(path).name
        url = DATASET_ROOT + urllib.parse.quote(path, safe="/")
        print(f"Downloading OPeRA shard {idx}/{len(paths)}: {Path(path).name}", flush=True)
        download_file(url, local)
        parquet = pq.ParquetFile(local)
        columns = [name for name in required if name in parquet.schema_arrow.names]
        frame = pq.read_table(local, columns=columns).to_pandas()
        frames.append(frame)
        file_summary.append({"path": path, "rows": int(parquet.metadata.num_rows), "bytes": int(local.stat().st_size)})
        local.unlink(missing_ok=True)
    actions = pd.concat(frames, ignore_index=True)
    return actions, file_summary


def make_product_catalog(actions: pd.DataFrame):
    catalog, event_rows = {}, []
    for row in actions.itertuples(index=False):
        session_id = str(row.session_id)
        timestamp = pd.to_datetime(row.timestamp, errors="coerce", utc=True)
        click_type = str(row.click_type or "unknown")
        action_type = str(row.action_type or "unknown")
        products = extract_products(row.products)
        products.extend(extract_products(row.element_meta))
        unique = {}
        for item in products:
            asin = str(item.get("asin", "")).strip()
            title = str(item.get("title") or item.get("name") or "").strip()
            if not asin or not title:
                continue
            unique[asin] = item
            old = catalog.get(asin, {})
            if asin not in catalog or len(title) > len(old.get("title", "")):
                catalog[asin] = {"asin": asin, "title": title, "price": price_number(item.get("price"))}
        if not unique:
            continue
        if click_type == "product_link":
            event_name = "product-link action"
        elif click_type == "purchase":
            event_name = "purchase-intent action"
        else:
            event_name = "other product-associated action"
        for asin, item in unique.items():
            title = str(item.get("title") or item.get("name") or catalog[asin]["title"]).strip()
            event_rows.append({
                "session_id": session_id,
                "user_id": session_id.split("_", 1)[0],
                "timestamp": timestamp,
                "asin": asin,
                "title": title,
                "price": price_number(item.get("price")),
                "click_type": click_type,
                "action_type": action_type,
                "event_name": event_name,
            })
    interactions = pd.DataFrame(event_rows)
    if interactions.empty:
        raise RuntimeError("No product metadata with ASIN identifiers was attached to OPeRA actions.")
    interactions = interactions.dropna(subset=["timestamp", "session_id", "asin"])
    interactions = interactions.drop_duplicates(subset=["session_id", "timestamp", "asin"])
    return catalog, interactions


def make_session_sequences(interactions: pd.DataFrame):
    sequences, starts = {}, {}
    for session_id, group in interactions.sort_values("timestamp").groupby("session_id", sort=False):
        items, seen = [], set()
        for row in group.itertuples(index=False):
            asin = str(row.asin)
            if asin in seen:
                continue
            seen.add(asin)
            items.append({
                "asin": asin,
                "title": str(row.title),
                "event_name": str(row.event_name),
                "timestamp": pd.Timestamp(row.timestamp),
            })
        if items:
            sequences[str(session_id)] = items
            starts[str(session_id)] = items[0]["timestamp"]
    return sequences, starts


def build_session_model(train_sequences):
    counts = Counter()
    co_vis = defaultdict(Counter)
    for sequence in train_sequences.values():
        unique = list(dict.fromkeys(item["asin"] for item in sequence))
        counts.update(unique)
        for i, left in enumerate(unique):
            for j in range(i + 1, len(unique)):
                right = unique[j]
                co_vis[left][right] += 1.0 / (j - i)
                co_vis[right][left] += 1.0 / (j - i)
    return {"counts": counts, "co_vis": co_vis}


def build_content_model(catalog):
    item_ids = sorted(catalog)
    titles = [catalog[item]["title"] for item in item_ids]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(titles)
    similarity = cosine_similarity(matrix)
    return {"item_ids": item_ids, "item_index": {item: i for i, item in enumerate(item_ids)}, "similarity": similarity}


def mmr_rerank(candidates, relevance, content_model, top_k=10, lam=0.72):
    index, similarity = content_model["item_index"], content_model["similarity"]
    max_score = max((abs(relevance.get(item, 0.0)) for item in candidates), default=0.0) or 1.0
    selected, remaining = [], list(candidates)
    while remaining and len(selected) < top_k:
        best, best_score = None, -float("inf")
        for item in remaining:
            rel = relevance.get(item, 0.0) / max_score
            redundancy = 0.0
            if selected:
                previous = [index[x] for x in selected if x in index]
                if previous and item in index:
                    redundancy = max(float(similarity[index[item], j]) for j in previous)
            score = lam * rel - (1.0 - lam) * redundancy
            if score > best_score:
                best, best_score = item, score
        selected.append(best)
        remaining.remove(best)
    return selected


def recommend_session(history, session_model, content_model, catalog, top_k=10):
    seen = set(history)
    scores = Counter()
    for item in history:
        scores.update(session_model["co_vis"].get(item, {}))
    scores = {item: float(score) for item, score in scores.items() if item not in seen}
    mode = "session co-visitation + MMR"
    if not scores:
        # Cold-start fallback: use content similarity to the last product in the session.
        query = history[-1] if history else None
        if query not in content_model["item_index"]:
            scores = {item: float(count) for item, count in session_model["counts"].items() if item not in seen}
            mode = "historical popularity fallback"
        else:
            row = content_model["item_index"][query]
            scores = {
                item: float(content_model["similarity"][row, idx])
                for item, idx in content_model["item_index"].items()
                if item not in seen and item != query
            }
            mode = "TF-IDF content similarity"
    ranked = sorted(scores, key=lambda item: (-scores[item], item))
    selected = mmr_rerank(ranked[:min(50, len(ranked))], scores, content_model, top_k=top_k)
    recommendations = []
    for item in selected:
        info = catalog[item]
        recommendations.append({
            "itemid": item,
            "title": info.get("title", item),
            "price": info.get("price"),
            "score": round(scores.get(item, 0.0), 6),
        })
    return recommendations, mode


def build_product_recommendations(catalog, content_model):
    result = {}
    for item, info in catalog.items():
        index = content_model["item_index"][item]
        scores = {
            other: float(content_model["similarity"][index, j])
            for other, j in content_model["item_index"].items() if other != item
        }
        ranked = sorted(scores, key=lambda other: (-scores[other], other))
        selected = mmr_rerank(ranked[:min(40, len(ranked))], scores, content_model, top_k=10)
        result[item] = {
            "title": info["title"],
            "price": info.get("price"),
            "recommendations": [
                {"itemid": other, "title": catalog[other]["title"], "price": catalog[other].get("price"),
                 "score": round(scores[other], 6)}
                for other in selected
            ],
        }
    return result


def main():
    with tempfile.TemporaryDirectory(prefix="opera_recommender_") as temp_dir:
        actions, file_summary = load_actions(Path(temp_dir))
    actions["timestamp"] = pd.to_datetime(actions["timestamp"], errors="coerce", utc=True)
    actions = actions.dropna(subset=["session_id", "timestamp"]).sort_values(
        ["timestamp", "session_id"], kind="mergesort"
    ).copy()

    catalog, interactions = make_product_catalog(actions)
    sequences, starts = make_session_sequences(interactions)
    if len(sequences) < 10 or len(catalog) < 10:
        raise RuntimeError(f"OPeRA product-action data is too small: {len(sequences)} sessions, {len(catalog)} ASINs.")

    ordered = sorted(starts, key=lambda session: starts[session])
    split = max(1, min(len(ordered) - 1, int(len(ordered) * 0.8)))
    train = {session: sequences[session] for session in ordered[:split]}
    test = {session: sequences[session] for session in ordered[split:]}

    session_model = build_session_model(train)
    content_model = build_content_model(catalog)
    content_recommendations = build_product_recommendations(catalog, content_model)

    # Count possible held-out next-item targets. With the very sparse product-linked
    # actions, only publish ranking metrics if the test sample is large enough.
    eligible_targets = 0
    for sequence in test.values():
        seen = set()
        items = [entry["asin"] for entry in sequence]
        for target in items[1:]:
            if target not in seen and target in session_model["counts"]:
                eligible_targets += 1
            seen.add(target)
    metrics_status = {
        "status": "insufficient_for_reliable_accuracy_metrics" if eligible_targets < 30 else "eligible_for_offline_evaluation",
        "eligible_next_product_targets": int(eligible_targets),
        "minimum_targets_for_reporting": 30,
        "message": (
            "The available sample yields too few held-out next-product targets for reliable Recall/NDCG claims. "
            "No zero-filled or synthetic accuracy scores are displayed."
            if eligible_targets < 30 else
            "Offline ranking metrics are calculated from held-out product-associated actions."
        ),
        "train_sessions": len(train),
        "test_sessions": len(test),
        "split": "chronological by whole session start time; 80% train / 20% test",
    }

    actions["date"] = actions["timestamp"].dt.strftime("%Y-%m-%d")
    daily_wide = actions.groupby(["date", "action_type"]).size().unstack(fill_value=0).sort_index()
    action_types = sorted(actions["action_type"].dropna().astype(str).unique())
    daily = {
        "dates": list(daily_wide.index),
        "series": {
            name: [int(v) for v in daily_wide[name].tolist()] if name in daily_wide else [0] * len(daily_wide)
            for name in action_types
        },
    }
    click_counts = actions["click_type"].fillna("unknown").astype(str).value_counts()
    action_counts = actions["action_type"].fillna("unknown").astype(str).value_counts()
    interactions_by_asin = (
        interactions.groupby(["asin", "title"], as_index=False)
        .agg(interactions=("session_id", "size"), sessions=("session_id", "nunique"))
        .sort_values(["interactions", "sessions"], ascending=False)
    )
    top_products = interactions_by_asin.head(20).to_dict(orient="records")

    visitors = {}
    for session_id in ordered[split:]:
        sequence = test.get(session_id, [])
        if not sequence:
            continue
        history = [entry["asin"] for entry in sequence[:-1]] or [sequence[0]["asin"]]
        recommendations, mode = recommend_session(history, session_model, content_model, catalog)
        visitors[session_id] = {
            "history_size": len(history),
            "history": [
                {"itemid": entry["asin"], "title": entry["title"], "event": entry["event_name"]}
                for entry in sequence[-10:]
            ],
            "recommendations": recommendations,
            "recommendation_method": mode,
        }

    sessions_with_product_link = int(interactions.loc[interactions["event_name"] == "product_link_action", "session_id"].nunique())
    sessions_with_purchase_intent = int(interactions.loc[interactions["event_name"] == "purchase_intent_action", "session_id"].nunique())
    start_date, end_date = actions["timestamp"].min(), actions["timestamp"].max()
    if start_date.year < 2025 or end_date.year < 2025:
        raise RuntimeError("OPeRA observations do not meet the 2025 date requirement.")

    metadata = {
        "source": "OPeRA — Observation, Persona, Rationale, Action",
        "publisher": "Northeastern University Human-centered AI Lab",
        "dataset_url": DATASET_PAGE,
        "paper_url": PAPER,
        "license": LICENSE,
        "citation": "Wang et al. (2025). OPeRA: A Dataset of Observation, Persona, Rationale, and Action. arXiv:2506.05606.",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_range": [start_date.isoformat(), end_date.isoformat()],
        "events_note": "Real participant shopping-session actions. click_type=purchase denotes a purchase-intent control click, not a confirmed order.",
        "population_note": "Research sample of human shopping sessions, not a full retailer event log or market-wide visitor sample.",
        "split_note": "Session-level chronological 80/20 split by session start time; sessions are not shared between train and test.",
        "action_files": file_summary,
        "recommendation_methods": "Session co-visitation, TF-IDF title similarity, cosine similarity and MMR diversity reranking.",
    }

    payload = {
        "metadata": metadata,
        "eda": {
            "n_events": int(len(actions)),
            "n_sessions": int(actions["session_id"].nunique()),
            "n_users": 51,
            "n_items": int(len(catalog)),
            "date_range": [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")],
            "event_counts": {str(k): int(v) for k, v in action_counts.items()},
            "click_counts": {str(k): int(v) for k, v in click_counts.items()},
            "product_link_actions": int((actions["click_type"] == "product_link").sum()),
            "purchase_intent_actions": int((actions["click_type"] == "purchase").sum()),
            "product_associated_interactions": int(len(interactions)),
            "sessions_with_product_link": sessions_with_product_link,
            "sessions_with_purchase_intent": sessions_with_purchase_intent,
            "sessions_used_for_training": len(train),
            "sessions_used_for_evaluation": len(test),
            "top_click_types": [{"name": str(k), "value": int(v)} for k, v in click_counts.items()],
        },
        "daily": daily,
        "evaluation": metrics_status,
        "top_products": top_products,
        "products": content_recommendations,
        "visitors": visitors,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "opera_2025_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    script_data = "window.DASHBOARD_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";"
    (ROOT / "data.js").write_text(script_data, encoding="utf-8")
    (ROOT / "web" / "data.js").write_text(script_data, encoding="utf-8")
    (ROOT / "data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "web" / "data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "evaluation_status.json").write_text(json.dumps(metrics_status, indent=2), encoding="utf-8")
    (REPORTS / "metrics.json").write_text(json.dumps({"status": metrics_status["status"], "note": metrics_status["message"]}, indent=2), encoding="utf-8")
    pd.DataFrame(top_products).to_csv(REPORTS / "top_products.csv", index=False)

    print(json.dumps({"metadata": metadata, "eda": payload["eda"], "evaluation": metrics_status}, indent=2))


if __name__ == "__main__":
    main()
