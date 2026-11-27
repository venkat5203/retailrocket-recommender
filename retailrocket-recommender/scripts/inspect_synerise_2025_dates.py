from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "synerise_2025_date_assessment.json"
ARCHIVE_URL = "https://data.recsys.synerise.com/dataset/challenge_dataset.tar.gz"
EVENT_FILES = {"product_buy", "add_to_cart", "remove_from_cart", "page_visit", "search_query"}


def iso(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    return str(value)


def inspect_parquet(path: Path, member_name: str):
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    timestamp_col = next((n for n in names if n.lower() in {"timestamp", "event_timestamp", "event_time"}), None)
    if timestamp_col is None:
        return {
            "file": member_name,
            "rows": parquet.metadata.num_rows,
            "columns": names,
            "timestamp_column": None,
            "date_range": None,
            "status": "timestamp_column_not_found",
        }

    col_index = names.index(timestamp_col)
    minimum = maximum = None
    stats_groups = 0
    for row_group_idx in range(parquet.metadata.num_row_groups):
        column = parquet.metadata.row_group(row_group_idx).column(col_index)
        stats = column.statistics
        if stats is None or not stats.has_min_max:
            continue
        try:
            low, high = stats.min, stats.max
            if minimum is None or low < minimum:
                minimum = low
            if maximum is None or high > maximum:
                maximum = high
            stats_groups += 1
        except TypeError:
            pass

    method = "parquet_row_group_statistics"
    if stats_groups == 0:
        method = "streamed_timestamp_column"
        minimum = maximum = None
        for batch in parquet.iter_batches(columns=[timestamp_col], batch_size=200_000):
            values = batch.column(0).to_pylist()
            for value in values:
                if value is None:
                    continue
                if minimum is None or value < minimum:
                    minimum = value
                if maximum is None or value > maximum:
                    maximum = value

    return {
        "file": member_name,
        "rows": parquet.metadata.num_rows,
        "columns": names,
        "timestamp_column": timestamp_col,
        "date_range": {"min": iso(minimum), "max": iso(maximum)},
        "date_range_method": method,
        "row_groups_with_timestamp_statistics": stats_groups,
    }


def main():
    with tempfile.TemporaryDirectory(prefix="synerise_date_check_") as temp_dir:
        temp = Path(temp_dir)
        archive = temp / "challenge_dataset.tar.gz"
        request = urllib.request.Request(
            ARCHIVE_URL,
            headers={"User-Agent": "VenkatRetailRocketDateAudit/1.0"},
        )
        with urllib.request.urlopen(request, timeout=180) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)

        results = {}
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith(".parquet"):
                    continue
                stem = Path(member.name).stem.lower()
                event_type = next((name for name in EVENT_FILES if name in stem), None)
                if event_type is None:
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                local_path = temp / (event_type + ".parquet")
                with source, local_path.open("wb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                results[event_type] = inspect_parquet(local_path, member.name)
                local_path.unlink(missing_ok=True)

        valid = [v["date_range"] for v in results.values() if v.get("date_range")]
        overall = {
            "min": min((v["min"] for v in valid if v.get("min")), default=None),
            "max": max((v["max"] for v in valid if v.get("max")), default=None),
        }
        report = {
            "assessed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "Synerise Universal Behavioral Modeling Dataset — RecSys 2025 Challenge",
            "download_url": ARCHIVE_URL,
            "official_dataset_page": "https://recsys.synerise.com/data-set",
            "license": "CC BY-NC 4.0",
            "archive_bytes": archive.stat().st_size,
            "event_files_found": sorted(results),
            "overall_event_date_range": overall,
            "events": results,
            "data_note": "The range is measured directly from Parquet event timestamps. Challenge publication date is not used as a substitute for event dates.",
        }
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
