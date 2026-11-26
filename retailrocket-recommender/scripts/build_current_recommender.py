"""Create 2026-updated product-catalogue recommendations from Open Food Facts.

Recommendations are content-based; the dataset contains no customer click, basket,
order, or purchase events.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import time
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://world.openfoodfacts.org/api/v2/search"
SOURCE_URL = "https://world.openfoodfacts.org/"
CUTOFF = datetime(2026, 1, 1, tzinfo=timezone.utc)
CUTOFF_EPOCH = int(CUTOFF.timestamp())
PAGE_SIZE = 100
MAX_PAGES = 3
REQUEST_GAP_SECONDS = 7.5
FIELDS = ",".join(["code", "product_name", "brands", "categories_tags_en", "quantity",
                   "last_modified_t", "ingredients_text", "nutrition_grades", "image_url",
                   "url", "nutriments"])


def get_page(session, page):
    params = {"page": page, "page_size": PAGE_SIZE, "sort_by": "last_modified_t", "fields": FIELDS}
    for _ in range(4):
        response = session.get(API_URL, params=params, timeout=45)
        if response.status_code in (429, 503):
            wait = int(response.headers.get("Retry-After", "30"))
            time.sleep(min(max(wait, 15), 120))
            continue
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload.get("products"), list):
            raise RuntimeError("Open Food Facts did not return a products list.")
        return payload
    raise RuntimeError("Open Food Facts rate limit persisted after retries.")


def iso(ts):
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def normalize(p):
    tags = p.get("categories_tags_en") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    brands = str(p.get("brands") or "").strip()
    code = str(p.get("code") or "").strip()
    nutriments = p.get("nutriments") or {}
    product_url = str(p.get("url") or "")
    if not product_url.startswith("https://"):
        product_url = "https://world.openfoodfacts.org/product/" + code
    return {
        "barcode": code,
        "name": str(p.get("product_name") or "").strip(),
        "brand": brands.split(",")[0].strip() if brands else "",
        "brands": brands,
        "category": str(tags[0]).replace("-", " ").title() if tags else "Uncategorized",
        "categories": [str(x).replace("-", " ").title() for x in tags],
        "quantity": str(p.get("quantity") or ""),
        "ingredients": str(p.get("ingredients_text") or ""),
        "nutrition_grade": str(p.get("nutrition_grades") or ""),
        "last_modified_t": int(p.get("last_modified_t") or 0),
        "last_modified_at_utc": iso(p.get("last_modified_t")),
        "image_url": str(p.get("image_url") or ""),
        "product_url": product_url,
        "energy_kcal_100g": nutriments.get("energy-kcal_100g"),
        "sugars_100g": nutriments.get("sugars_100g"),
        "fat_100g": nutriments.get("fat_100g"),
        "proteins_100g": nutriments.get("proteins_100g"),
    }


def collect():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "VenkatRetailRocketContentRecommender/2.0 (venkata.nemani99@gmail.com)",
        "Accept": "application/json",
    })
    first = get_page(session, 1)
    page_count = max(1, math.ceil(int(first.get("count") or 0) / PAGE_SIZE))
    sample = first.get("products", [])
    stamps = [int(p.get("last_modified_t") or 0) for p in sample]
    descending = not stamps or stamps[0] >= stamps[-1]
    pages = list(range(1, min(MAX_PAGES, page_count) + 1)) if descending else list(range(max(1, page_count - MAX_PAGES + 1), page_count + 1))
    raw = []
    for page in pages:
        payload = first if page == 1 else None
        if payload is None:
            time.sleep(REQUEST_GAP_SECONDS)
            payload = get_page(session, page)
        raw.extend(payload.get("products", []))

    unique = {}
    for p in raw:
        code = str(p.get("code") or "").strip()
        stamp = int(p.get("last_modified_t") or 0)
        if code and stamp >= CUTOFF_EPOCH:
            if code not in unique or stamp > int(unique[code].get("last_modified_t") or 0):
                unique[code] = p
    if len(unique) < 3:
        newest = iso(max(stamps, default=0)) or "unavailable"
        raise RuntimeError("Fewer than 3 products updated since 2026-01-01. Latest timestamp on first page: " + newest + ". No historical interaction data or fabricated products were substituted.")
    return list(unique.values()), {
        "count_reported_by_api": int(first.get("count") or 0),
        "pages_requested": pages,
        "page_size": PAGE_SIZE,
        "raw_records_received": len(raw),
        "unique_products_updated_since_2026": len(unique),
        "sort_direction_detected": "descending" if descending else "ascending",
    }


def make_recommendations(products):
    documents = []
    for p in products:
        documents.append(" ".join([
            p["name"], p["name"], p["brand"], p["brand"], p["category"],
            " ".join(p["categories"]), p["quantity"], p["ingredients"], p["nutrition_grade"]
        ]).lower())

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=20000)
    matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(matrix, dense_output=True)
    result = {}
    for i, product in enumerate(products):
        candidates = [j for j in range(len(products)) if j != i and similarity[i, j] > 0]
        candidates = sorted(candidates, key=lambda j: similarity[i, j], reverse=True)[:60]
        selected = []
        while candidates and len(selected) < 10:
            best_index = None
            best_value = -1.0
            for j in candidates:
                redundancy = max([float(similarity[j, chosen]) for chosen in selected] or [0.0])
                value = 0.72 * float(similarity[i, j]) - 0.28 * redundancy
                if value > best_value:
                    best_index, best_value = j, value
            if best_index is None:
                break
            selected.append(best_index)
            candidates.remove(best_index)

        recs = []
        for j in selected:
            rec = dict(products[j])
            rec["similarity"] = round(float(similarity[i, j]), 4)
            recs.append(rec)
        result[product["barcode"]] = recs
    return result


def main():
    print("Fetching Open Food Facts products whose last-modified date is in 2026...")
    raw, page_info = collect()
    products = [normalize(p) for p in raw]
    products.sort(key=lambda p: p["last_modified_t"], reverse=True)
    recommendations = make_recommendations(products)

    category_counts = {}
    brand_counts = {}
    for p in products:
        category_counts[p["category"]] = category_counts.get(p["category"], 0) + 1
        if p["brand"]:
            brand_counts[p["brand"]] = brand_counts.get(p["brand"], 0) + 1

    categories = [{"category": k, "count": v} for k, v in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:15]]
    brands = [{"brand": k, "count": v} for k, v in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:15]]
    dates = [p["last_modified_at_utc"] for p in products if p["last_modified_at_utc"]]
    payload = {
        "metadata": {
            "source": "Open Food Facts",
            "source_url": SOURCE_URL,
            "api_url": API_URL,
            "license": "Open Database License (ODbL) 1.0",
            "attribution": "Open Food Facts contributors",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "required_last_modified_from_utc": CUTOFF.isoformat(),
            "earliest_product_update_utc": min(dates) if dates else "",
            "latest_product_update_utc": max(dates) if dates else "",
            "page_info": page_info,
            "method": "TF-IDF content similarity using product name, brand, category, quantity, ingredients and nutrition grade; MMR reranking with lambda=0.72.",
            "limitation": "No real customer clicks, baskets, orders or purchases are available here. Recommendations are content-based, not behavioral personalization.",
        },
        "summary": {
            "products": len(products),
            "categories": len(category_counts),
            "brands": len(brand_counts),
            "recommendation_sets": len(recommendations),
        },
        "categories": categories,
        "brands": brands,
        "products": products,
        "recommendations": recommendations,
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    js = "window.DASHBOARD_DATA = " + serialized.replace("</", "<\\/") + ";\n"
    (ROOT / "data.js").write_text(js, encoding="utf-8")
    web = ROOT / "web"
    web.mkdir(parents=True, exist_ok=True)
    (web / "data.js").write_text(js, encoding="utf-8")

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "open_food_facts_2026_snapshot.json").write_text(
        json.dumps({"retrieved_at_utc": payload["metadata"]["retrieved_at_utc"], "page_info": page_info, "products": raw}, ensure_ascii=False),
        encoding="utf-8",
    )
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "current_catalogue_summary.json").write_text(
        json.dumps(payload["metadata"] | payload["summary"], indent=2), encoding="utf-8"
    )
    print(json.dumps({"metadata": payload["metadata"], "summary": payload["summary"], "top_categories": categories[:5]}, indent=2))


if __name__ == "__main__":
    main()
