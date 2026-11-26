#!/usr/bin/env python3
"""Check whether real zero-inflated continuous dataset files are present and intact.

Runs without third-party Python packages.
From repository root: python data/zero_inflated_papers/verify_datasets.py
"""
import csv
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = {
    "saeczi/raw/samp.rda": ("6df7eb79d82f62777e2fe2f4fa02a18d7bf76bd3", 27899),
    "saeczi/raw/pop.rda": ("e42b6a1ac2484069f0d2fadeca5ddd1b58cf74df", 124974),
    "rainfallbench/stations/J340_data.csv": ("e18bf3ff1125621d824c0f41bde9907610f9feaa", 4652001),
    "rainfallbench/stations/ZIMM_data.csv": ("125f068f4f8b9effa8016614e2d7543ebb01ff8d", 4680618),
    "rainfallbench/stations/P095_data.csv": ("40fdf4fec2f56ea7aab3b200e41fbb20301e0aee", 4536152),
    "hybrid_nhmm_copula/raw/mapa_estaciones_precipitaciones_clean.csv.zip": ("0d007e8f179c4602ca5d44a294cf5c2defda623f", 33960384),
}
FALLBACKS = {
    "rainfallbench/stations/J340_data.csv": "rainfallbench/upstream/dataset/gnss/Asia/J340_data.csv",
    "rainfallbench/stations/ZIMM_data.csv": "rainfallbench/upstream/dataset/gnss/Europe/ZIMM_data.csv",
    "rainfallbench/stations/P095_data.csv": "rainfallbench/upstream/dataset/gnss/North_America/P095_data.csv",
}


def git_blob_sha(path):
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(("blob %d\0" % size).encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zero_stats(path):
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = csv.DictReader(f)
            cols = rows.fieldnames or []
            possible = [c for c in ("gpm_precip_sum_1h", "tp", "tp_era5", "precipitation", "rainfall", "rain") if c in cols]
            if not possible:
                return {"columns": cols, "warning": "No known precipitation target column; requires schema inspection"}
            col = possible[0]
            total = zeros = positives = 0
            unique_nonzero = set()
            for row in rows:
                try:
                    v = float(row[col])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(v):
                    continue
                total += 1
                if v == 0:
                    zeros += 1
                if v > 0:
                    positives += 1
                    if len(unique_nonzero) < 5000:
                        unique_nonzero.add(v)
            return {
                "target": col, "valid_rows": total, "zero_rows": zeros,
                "positive_rows": positives,
                "zero_ratio": zeros / total if total else None,
                "distinct_positive_values_sampled": len(unique_nonzero),
                "columns": cols,
            }
    except (UnicodeDecodeError, OSError) as exc:
        return {"error": str(exc)}


def verify():
    result = {}
    for expected_path, (expected_sha, expected_size) in FILES.items():
        actual = ROOT / expected_path
        source = "committed_or_submodule"
        if not actual.is_file() and expected_path in FALLBACKS:
            fallback = ROOT / FALLBACKS[expected_path]
            if fallback.is_file():
                actual, source = fallback, "author_submodule"
        if not actual.is_file():
            result[expected_path] = {"status": "missing", "note": "Run git submodule update --init --recursive if applicable"}
            continue
        size = actual.stat().st_size
        sha = git_blob_sha(actual)
        info = {
            "status": "verified" if (size == expected_size and sha == expected_sha) else "mismatch",
            "source": source, "local_path": str(actual.relative_to(ROOT)),
            "bytes": size, "git_blob_sha1": sha,
        }
        if info["status"] == "verified" and actual.suffix.lower() == ".csv":
            info["statistics"] = zero_stats(actual)
        if info["status"] == "verified" and actual.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(actual) as z:
                    info["zip_contents"] = [
                        {"name": x.filename, "uncompressed_bytes": x.file_size}
                        for x in z.infolist()[:12]
                    ]
                    if z.testzip() is not None:
                        info["status"] = "zip_crc_error"
            except (zipfile.BadZipFile, OSError) as exc:
                info["status"] = "zip_error"
                info["error"] = str(exc)
        result[expected_path] = info
    return result


if __name__ == "__main__":
    report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if "--json" in sys.argv:
        pos = sys.argv.index("--json")
        if pos + 1 >= len(sys.argv):
            raise SystemExit("--json requires a file path")
        dest = Path(sys.argv[pos + 1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(x["status"] not in ("verified", "missing") for x in report.values()):
        raise SystemExit(1)
