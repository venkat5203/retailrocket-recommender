from __future__ import annotations

import json
import math
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
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
API = "https://huggingface.co/api/datasets/NEU-HAI/OPeRA/tree/main/OPeRA_full/action?recursive=true&expand=false"
DATASET_ROOT = "https://huggingface.co/datasets/NEU-HAI/OPeRA/resolve/main/"
DATASET_PAGE = "https://huggingface.co/datasets/NEU-HAI/OPeRA"
PAPER = "https://arxiv.org/abs/2506.05606"
LICENSE = "CC BY 4.0"
TOP_K = 10


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
    """Walk product metadata fields and return product dictionaries with ASIN identifiers."""
    if isinstance(value, str):
        decoded = _decode_json(value)
        if decoded is None:
            return []
        return extract_products(decoded)
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
        if item.get("type") == "file"
        and item.get("path", "").endswith(".parquet")
        and ("/train-" in item.get("path", "") or "/test-" in item.get("path", ""))
    )
    if not paths:
        raise RuntimeError("No OPeRA full-action Parquet shards were found.")

    required = ["session_id", "timestamp", "action_type", "click_type", "semantic_id", "element_meta", "products"]
    frames = []
    file_summary = []
    for idx, path in enumerate(paths, start=1):
        local = temp_dir / Path(path).name
        url = DATASET_ROOT + urllib.parse.quote(path, safe="/")
        print(f"Downloading OPeRA Parquet shard {idx}/{len(paths)}: {Path(path).name}", flush=True)
        download_file(url, local)
        parquet = pq.ParquetFile(local)
        available = parquet.schema_arrow.names
        columns = [name for name in required if name in available]
        table = pq.read_table(local, columns=columns)
        frame = table.to_pandas()
        if not frame.empty:
            frames.append(frame)
        file_summary.append({
            "path": path,
            "rows": int(parquet.metadata.num_rows),
            "bytes": int(local.stat().st_size),
        })
        local.unlink(missing_ok=True)

    if not frames:
        raise RuntimeError("OPeRA action files downloaded but contained no rows.")
    actions = pd.concat(frames, ignore_index=True)
    return actions, file_summary


def make_product_catalog(actions: pd.DataFrame):
    catalog = {}
    rows = []
    for row in actions.itertuples(index=False):
        session_id = str(row.session_id)
        timestamp = pd.to_datetime(row.timestamp, errors="coerce", utc=True)
        action_type = str(row.action_type or "unknown")
        click_type = str(row.click_type or "unknown")

        products = extract_products(row.products)
        products.extend(extract_products(row.element_meta))
        unique = {}
        for item in products:
            asin = str(item.get("asin", "")).strip()
            title = str(item.get("title") or item.get("name") or "").strip()
            if asin and title:
                unique[asin] = item
                if asin not in catalog or len(title) > len(catalog[asin].get("title", "")):
                    catalog[asin] = {
                        "asin": asin,
                        "title": title,
                        "price": price_number(item.get("price")),
                    }

        if unique:
            if click_type == "product_link":
                event_name, weight = "product_link_action", 1.0
            elif click_type == "purchase":
                event_name, weight = "purchase_intent_action", 2.0
            else:
                event_name, weight = "product_associated_action", 0.5

            for asin, item in unique.items():
                title = str(item.get("title") or item.get("name") or catalog[asin]["title"]).strip()
                rows.append({
                    "session_id": session_id,
                    "user_id": session_id.split("_", 1)[0],
                    "timestamp": timestamp,
                    "asin": asin,
                    "title": title,
                    "price": price_number(item.get("price")),
                    "click_type": click_type,
                    "action_type": action_type,
                    "event_name": event_name,
                    "weight": weight,
                })

    interactions = pd.DataFrame(rows)
    if interactions.empty:
        raise RuntimeError("No product records with ASIN identifiers were attached to the OPeRA actions.")
    interactions = interactions.dropna(subset=["timestamp", "session_id", "asin"])
    interactions = interactions.drop_duplicates(subset=["session_id", "timestamp", "asin", "event_name"])
    return catalog, interactions


def make_session_sequences(interactions: pd.DataFrame):
    sequences = {}
    starts = {}
    for session_id, group in interactions.sort_values("timestamp").groupby("session_id", sort=False):
        ordered = []
        for event in group.itertuples(index=False):
            asin = str(event.asin)
            if not ordered or ordered[-1]["asin"] != asin:
                ordered.append({
                    "asin": asin,
                    "title": str(event.title),
                    "event_name": str(event.event_name),
                    "timestamp": pd.Timestamp(event.timestamp),
                })
        unique_items, seen = [], set()
        for item in ordered:
            if item["asin"] not in seen:
                unique_items.append(item)
                seen.add(item["asin"])
        if unique_items:
            sequences[str(session_id)] = unique_items
            starts[str(session_id)] = unique_items[0]["timestamp"]
    return sequences, starts


def train_models(train_sequences, catalog):
    counts = Counter()
    co_vis = defaultdict(Counter)
    matrix_rows = []
    item_set = set()

    for sequence in train_sequences.values():
        items = [entry["asin"] for entry in sequence]
        item_set.update(items)
        counts.update(items)
        unique = list(dict.fromkeys(items))
        for i, left in enumerate(unique):
            for j in range(i + 1, len(unique)):
                right = unique[j]
                co_vis[left][right] += 1.0 / (j - i)
                co_vis[right][left] += 1.0 / (j - i)
        matrix_rows.append(unique)

    item_ids = sorted(item_set)
    item_index = {item: idx for idx, item in enumerate(item_ids)}
    session_ids = list(train_sequences)
    session_index = {session: idx for idx, session in enumerate(session_ids)}
    rows, columns, values = [], [], []
    for session, items in zip(session_ids, matrix_rows):
        for item in items:
            rows.append(session_index[session])
            columns.append(item_index[item])
            values.append(1.0)
    matrix = csr_matrix((values, (rows, columns)), shape=(len(session_ids), len(item_ids)), dtype=np.float32)

    svd = None
    item_factors = None
    if min(matrix.shape) >= 3 and matrix.nnz >= 8:
        components = max(2, min(24, min(matrix.shape) - 1))
        svd = TruncatedSVD(n_components=components, n_iter=7, random_state=42)
        svd.fit(matrix)
        item_factors = svd.components_.T.astype(np.float32)

    titles = [catalog.get(item, {}).get("title", item) for item in item_ids]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    try:
        text_matrix = vectorizer.fit_transform(titles)
        title_similarity = cosine_similarity(text_matrix)
    except ValueError:
        title_similarity = np.eye(len(item_ids), dtype=np.float32)

    return {
        "counts": counts,
        "co_vis": co_vis,
        "item_ids": item_ids,
        "item_index": item_index,
        "svd": svd,
        "item_factors": item_factors,
        "title_similarity": title_similarity,
    }


def recommendation_record(item, score, catalog):
    info = catalog.get(item, {})
    return {
        "itemid": str(item),
        "title": info.get("title", str(item)),
        "price": info.get("price"),
        "score": round(float(score), 6),
    }


def mmr_rerank(candidates, relevance, model, catalog, top_k, lam=0.72):
    index = model["item_index"]
    similarity = model["title_similarity"]
    if not candidates:
        return []
    max_score = max(abs(relevance.get(item, 0.0)) for item in candidates) or 1.0
    selected, remaining = [], list(candidates)
    while remaining and len(selected) < top_k:
        best_item, best_value = None, -float("inf")
        for item in remaining:
            rel = relevance.get(item, 0.0) / max_score
            redundancy = 0.0
            if selected and item in index:
                previous = [index[x] for x in selected if x in index]
                if previous:
                    redundancy = max(float(similarity[index[item], j]) for j in previous)
            value = lam * rel - (1.0 - lam) * redundancy
            if value > best_value:
                best_item, best_value = item, value
        selected.append(best_item)
        remaining.remove(best_item)
    return [recommendation_record(item, relevance.get(item, 0.0), catalog) for item in selected]


def rank_recommendations(model, history, catalog, mode="item_knn", top_k=10, mmr=True):
    seen = set(history)
    candidates = [item for item in model["item_ids"] if item not in seen]
    if not candidates:
        return []

    scores = {}
    if mode == "popularity":
        scores = {item: float(model["counts"].get(item, 0)) for item in candidates}
    elif mode == "item_knn":
        for source in history:
            for candidate, weight in model["co_vis"].get(source, {}).items():
                if candidate not in seen:
                    scores[candidate] = scores.get(candidate, 0.0) + float(weight)
    elif mode == "svd" and model["svd"] is not None:
        indices = [model["item_index"][item] for item in history if item in model["item_index"]]
        if indices:
            user_vector = model["item_factors"][indices].mean(axis=0)
            values = model["item_factors"] @ user_vector
            scores = {item: float(values[model["item_index"][item]]) for item in candidates}

    if not scores:
        scores = {item: float(model["counts"].get(item, 0)) for item in candidates}
    ranked = sorted(scores, key=lambda item: (-scores[item], item))
    if mmr and mode == "item_knn" and len(ranked) > 1:
        return mmr_rerank(ranked[:min(50, len(ranked))], scores, model, catalog, top_k)
    return [recommendation_record(item, scores[item], catalog) for item in ranked[:top_k]]


def evaluate_models(test_sequences, model, catalog):
    results = {name: {"hits": [], "ranks": [], "ranked": []} for name in ("popularity", "item_knn", "svd")}
    latencies = {name: [] for name in results}
    target_count = 0
    for sequence in test_sequences.values():
        items = [entry["asin"] for entry in sequence]
        for position in range(1, len(items)):
            target = items[position]
            history = items[:position]
            if target in history or target not in model["item_index"]:
                continue
            target_count += 1
            for name in results:
                start = time.perf_counter()
                recs = rank_recommendations(model, history, catalog, mode=name, top_k=TOP_K, mmr=(name == "item_knn"))
                latencies[name].append((time.perf_counter() - start) * 1000)
                ranked = [rec["itemid"] for rec in recs]
                rank = ranked.index(target) + 1 if target in ranked else None
                results[name]["hits"].append(1 if rank else 0)
                results[name]["ranks"].append(rank)
                results[name]["ranked"].append(ranked)

    if target_count == 0:
        raise RuntimeError("No leakage-safe next-product targets could be formed from OPeRA test sessions.")

    metrics = {}
    for name, result in results.items():
        hits, ranks = result["hits"], result["ranks"]
        ranked_lists = result["ranked"]
        metrics[name] = {
            "evaluated_targets": len(hits),
            "recall@10": round(float(np.mean(hits)), 6),
            "hit_rate@10": round(float(np.mean(hits)), 6),
            "ndcg@10": round(float(np.mean([1 / math.log2(rank + 1) if rank else 0.0 for rank in ranks])), 6),
            "map@10": round(float(np.mean([1 / rank if rank else 0.0 for rank in ranks])), 6),
            "catalog_coverage": round(len({item for row in ranked_lists for item in row}) / max(1, len(model["item_ids"])), 6),
            "mean_inference_latency_ms": round(float(np.mean(latencies[name])) if latencies[name] else 0.0, 4),
        }
    return metrics, target_count


def main():
    with tempfile.TemporaryDirectory(prefix="opera_recommender_") as temp_dir:
        actions, file_summary = load_actions(Path(temp_dir))

    actions["timestamp"] = pd.to_datetime(actions["timestamp"], errors="coerce", utc=True)
    actions = actions.dropna(subset=["session_id", "timestamp"]).sort_values(
        ["timestamp", "session_id"], kind="mergesort"
    ).copy()

    catalog, interactions = make_product_catalog(actions)
    sequences, session_starts = make_session_sequences(interactions)
    if len(sequences) < 10 or len(catalog) < 10:
        raise RuntimeError(
            f"OPeRA product-interaction sample too small: {len(sequences)} sessions, {len(catalog)} products."
        )

    ordered_sessions = sorted(session_starts, key=lambda sid: session_starts[sid])
    split_pos = max(1, min(len(ordered_sessions) - 1, int(len(ordered_sessions) * 0.8)))
    train_sessions = {sid: sequences[sid] for sid in ordered_sessions[:split_pos]}
    test_sessions = {sid: sequences[sid] for sid in ordered_sessions[split_pos:]}
    model = train_models(train_sessions, catalog)
    metrics, evaluated_targets = evaluate_models(test_sessions, model, catalog)

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
    product_link_sessions = interactions.loc[interactions["event_name"] == "product_link_action", "session_id"].nunique()
    purchase_intent_sessions = interactions.loc[interactions["event_name"] == "purchase_intent_action", "session_id"].nunique()

    top_products = (
        interactions.groupby(["asin", "title"], as_index=False)
        .agg(interactions=("session_id", "size"), sessions=("session_id", "nunique"))
        .sort_values(["interactions", "sessions"], ascending=False).head(20)
    )
    visitors = {}
    for session_id in list(test_sessions)[:40]:
        sequence = test_sessions[session_id]
        if len(sequence) < 2:
            continue
        history = sequence[:-1]
        history_ids = [entry["asin"] for entry in history]
        visitors[session_id] = {
            "history_size": len(history_ids),
            "history": [
                {"itemid": entry["asin"], "title": entry["title"], "event": entry["event_name"]}
                for entry in history[-10:]
            ],
            "held_out_next_item": {"itemid": sequence[-1]["asin"], "title": sequence[-1]["title"]},
            "recommendations": rank_recommendations(model, history_ids, catalog, mode="item_knn", top_k=10, mmr=True),
        }

    start_date, end_date = actions["timestamp"].min(), actions["timestamp"].max()
    if start_date.year < 2025 or end_date.year < 2025:
        raise RuntimeError("OPeRA observations do not meet the requested 2025 date requirement.")

    metadata = {
        "source": "OPeRA — Observation, Persona, Rationale, Action",
        "publisher": "Northeastern University Human-centered AI Lab",
        "dataset_url": DATASET_PAGE,
        "paper_url": PAPER,
        "license": LICENSE,
        "citation": "Wang et al. (2025). OPeRA: A Dataset of Observation, Persona, Rationale, and Action. arXiv:2506.05606.",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_range": [start_date.isoformat(), end_date.isoformat()],
        "events_note": "Real participant shopping-session actions; click_type=purchase denotes a purchase-intent control click, not a confirmed order.",
        "population_note": "A research sample of real human shopping sessions, not a complete retailer event log or a market-wide visitor sample.",
        "split_note": "Session-level chronological 80/20 split by session start time. Sessions do not cross train/evaluation partitions.",
        "action_files": file_summary,
    }
    payload = {
        "metadata": metadata,
        "eda": {
            "n_events": int(len(actions)),
            "n_sessions": int(actions["session_id"].nunique()),
            "n_users": int(actions["session_id"].astype(str).str.split("_", n=1).str[0].nunique()),
            "n_items": int(len(catalog)),
            "date_range": [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")],
            "event_counts": {str(k): int(v) for k, v in action_counts.items()},
            "click_counts": {str(k): int(v) for k, v in click_counts.items()},
            "product_link_actions": int((actions["click_type"] == "product_link").sum()),
            "purchase_intent_actions": int((actions["click_type"] == "purchase").sum()),
            "product_associated_interactions": int(len(interactions)),
            "sessions_with_product_link": int(product_link_sessions),
            "sessions_with_purchase_intent": int(purchase_intent_sessions),
            "sessions_used_for_training": len(train_sessions),
            "sessions_used_for_evaluation": len(test_sessions),
            "evaluated_next_product_targets": evaluated_targets,
            "top_click_types": [{"name": str(k), "value": int(v)} for k, v in click_counts.items()],
        },
        "daily": daily,
        "metrics": metrics,
        "top_products": top_products.to_dict(orient="records"),
        "visitors": visitors,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "opera_2025_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    script_data = "window.DASHBOARD_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";"
    (ROOT / "data.js").write_text(script_data, encoding="utf-8")
    (ROOT / "web" / "data.js").write_text(script_data, encoding="utf-8")
    (ROOT / "data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "web" / "data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([{"model": name, **scores} for name, scores in metrics.items()]).to_csv(REPORTS / "metrics.csv", index=False)
    print(json.dumps({"metadata": metadata, "eda": payload["eda"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
