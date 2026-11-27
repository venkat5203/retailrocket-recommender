from __future__ import annotations

import json
import tempfile
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

BASE = "https://huggingface.co"
DATASET = "NEU-HAI/OPeRA"
API = f"{BASE}/api/datasets/{DATASET}/tree/main/OPeRA_filtered/action?recursive=true&expand=false"
REPORT = Path(__file__).resolve().parents[1] / "reports" / "opera_2025_assessment.json"


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "VenkatRetailRecommenderDateAudit/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def main():
    files = get_json(API)
    parquet_files = [item for item in files if item.get("type") == "file" and item.get("path", "").endswith(".parquet")]
    if not parquet_files:
        raise RuntimeError("No OPeRA filtered-action Parquet files were found in the public dataset repository.")

    report = {
        "assessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "OPeRA — Observation, Persona, Rationale, Action",
        "publisher": "Northeastern University Human-centered AI Lab",
        "dataset_url": "https://huggingface.co/datasets/NEU-HAI/OPeRA",
        "paper": "https://arxiv.org/abs/2506.05606",
        "license": "CC BY 4.0",
        "action_files": [],
        "timestamp_range": {"min": None, "max": None},
        "aggregates": {"rows": 0, "sessions": 0, "action_types": {}, "click_types": {}, "product_rows": 0, "unique_asins": 0},
    }
    all_sessions = set()
    all_asins = set()
    actions = Counter()
    clicks = Counter()

    with tempfile.TemporaryDirectory(prefix="opera_2025_") as temp:
        for index, item in enumerate(parquet_files):
            path = item["path"]
            name = Path(path).name
            local = Path(temp) / name
            url = f"{BASE}/datasets/{DATASET}/resolve/main/{urllib.parse.quote(path, safe='/')}"
            req = urllib.request.Request(url, headers={"User-Agent": "VenkatRetailRecommenderDateAudit/1.0"})
            with urllib.request.urlopen(req, timeout=120) as response, local.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)

            parquet = pq.ParquetFile(local)
            columns = parquet.schema_arrow.names
            if "timestamp" not in columns:
                raise RuntimeError(f"{name}: timestamp column is missing")

            minimum = maximum = None
            timestamp_idx = columns.index("timestamp")
            for rg in range(parquet.metadata.num_row_groups):
                stats = parquet.metadata.row_group(rg).column(timestamp_idx).statistics
                if stats is not None and stats.has_min_max:
                    low, high = stats.min, stats.max
                    low, high = str(low), str(high)
                    if minimum is None or low < minimum:
                        minimum = low
                    if maximum is None or high > maximum:
                        maximum = high

            selected = [c for c in ["session_id", "timestamp", "action_type", "click_type", "semantic_id", "element_meta", "products"] if c in columns]
            table = pq.read_table(local, columns=selected)
            rows = table.to_pylist()
            all_sessions.update(str(row.get("session_id")) for row in rows if row.get("session_id") is not None)
            for row in rows:
                if row.get("action_type"):
                    actions[str(row["action_type"])] += 1
                if row.get("click_type"):
                    clicks[str(row["click_type"])] += 1
                products = row.get("products")
                if isinstance(products, str) and products.strip():
                    try:
                        obj = json.loads(products)
                        if isinstance(obj, dict):
                            obj = obj.get("products", obj.get("items", [obj]))
                        if isinstance(obj, list):
                            for product in obj:
                                if isinstance(product, dict) and product.get("asin"):
                                    all_asins.add(str(product["asin"]))
                    except (TypeError, ValueError):
                        pass

            if minimum and (report["timestamp_range"]["min"] is None or minimum < report["timestamp_range"]["min"]):
                report["timestamp_range"]["min"] = minimum
            if maximum and (report["timestamp_range"]["max"] is None or maximum > report["timestamp_range"]["max"]):
                report["timestamp_range"]["max"] = maximum
            report["aggregates"]["rows"] += len(rows)
            report["action_files"].append({
                "path": path,
                "rows": len(rows),
                "bytes": local.stat().st_size,
                "columns": columns,
                "timestamp_range": {"min": minimum, "max": maximum},
            })
            local.unlink(missing_ok=True)

    report["aggregates"]["sessions"] = len(all_sessions)
    report["aggregates"]["action_types"] = dict(actions)
    report["aggregates"]["click_types"] = dict(clicks)
    report["aggregates"]["unique_asins_in_products_field"] = len(all_asins)
    report["aggregates"]["product_rows"] = sum(clicks.values())
    report["data_limitations"] = [
        "Small observational sample (hundreds of sessions and dozens of participants); do not describe as millions of online retail visitors.",
        "Action logs represent observed browser interactions and product metadata; they are not a complete retailer transaction ledger.",
        "Purchases should only be claimed where an explicit purchase event or product quantity in the source record supports the claim."
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
