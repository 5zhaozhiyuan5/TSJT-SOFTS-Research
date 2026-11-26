#!/usr/bin/env python3
"""Verify three real continuous zero-inflated datasets in this Git branch.

The Chile dataset is a 216,657,356-byte CSV stored via Git LFS, NOT a zip.
Run 'git lfs pull' before this script if the CSV is a 134-byte pointer.
No external Python packages required.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIGINAL_BLOBS = {
    "saeczi/raw/samp.rda": ("6df7eb79d82f62777e2fe2f4fa02a18d7bf76bd3", 27899),
    "saeczi/raw/pop.rda": ("e42b6a1ac2484069f0d2fadeca5ddd1b58cf74df", 124974),
    "rainfallbench/stations/J340_data.csv": ("e18bf3ff1125621d824c0f41bde9907610f9feaa", 4652001),
    "rainfallbench/stations/ZIMM_data.csv": ("125f068f4f8b9effa8016614e2d7543ebb01ff8d", 4680618),
    "rainfallbench/stations/P095_data.csv": ("40fdf4fec2f56ea7aab3b200e41fbb20301e0aee", 4536152),
}
CHILE = "hybrid_nhmm_copula/raw/mapa_estaciones_precipitaciones_clean.csv"
CHILE_BYTES = 216657356
CHILE_SHA256 = "93609b0eacd2809f0d7bd86fa44ab0d54034faf7d306180fc02108307502d92a"


def sha_stream(path, algorithm, git_blob=False):
    h = hashlib.new(algorithm)
    if git_blob:
        h.update(("blob %s\0" % path.stat().st_size).encode())
    with path.open("rb") as f:
        for part in iter(lambda: f.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def inspect_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        target = next((x for x in ("gpm_precip_sum_1h", "precipitation", "tp", "tp_era5") if x in cols), None)
        if target is None:
            return {"columns": cols, "error": "No known precipitation target"}
        valid = zeros = positive = fractional_positive = 0
        for row in reader:
            try:
                value = float(row[target])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            valid += 1
            if value == 0:
                zeros += 1
            elif value > 0:
                positive += 1
                if not value.is_integer():
                    fractional_positive += 1
        return {
            "target": target, "valid_rows": valid, "zero_rows": zeros,
            "positive_rows": positive, "fractional_positive_rows": fractional_positive,
            "zero_ratio": zeros / valid if valid else None, "columns": cols
        }


def verify():
    report = {}
    for relative_path, (expected_hash, expected_size) in ORIGINAL_BLOBS.items():
        path = ROOT / relative_path
        if not path.is_file():
            report[relative_path] = {"status": "missing"}
            continue
        actual_hash = sha_stream(path, "sha1", git_blob=True)
        size = path.stat().st_size
        valid = actual_hash == expected_hash and size == expected_size
        details = {"status": "verified" if valid else "mismatch",
                   "bytes": size, "source_git_blob_sha1": actual_hash}
        if valid and path.suffix == ".csv":
            details["statistics"] = inspect_csv(path)
        report[relative_path] = details

    path = ROOT / CHILE
    if not path.is_file():
        report[CHILE] = {"status": "missing", "action": "git lfs pull"}
    elif path.stat().st_size < 512 and path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
        report[CHILE] = {
            "status": "lfs_pointer_only",
            "action": "Install Git LFS and run: git lfs pull",
            "bytes_on_disk": path.stat().st_size
        }
    else:
        actual_size = path.stat().st_size
        actual_sha = sha_stream(path, "sha256")
        valid = actual_size == CHILE_BYTES and actual_sha == CHILE_SHA256
        details = {"status": "verified" if valid else "mismatch",
                   "bytes": actual_size, "sha256": actual_sha}
        if valid:
            details["statistics"] = inspect_csv(path)
        report[CHILE] = details
    return report


if __name__ == "__main__":
    report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if "--json" in sys.argv:
        index = sys.argv.index("--json")
        if len(sys.argv) <= index + 1:
            raise SystemExit("--json requires output path")
        dest = Path(sys.argv[index + 1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(item["status"] != "verified" for item in report.values()):
        print("Some dataset files are missing or do not match expected checksums; see above.", file=sys.stderr)
        raise SystemExit(1)
