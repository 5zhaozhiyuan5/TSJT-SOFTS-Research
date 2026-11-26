#!/usr/bin/env python3
"""Continuously zero-inflated datasets: safe download/generation utilities.

From repo root:
  python data/zero_inflated/scripts/manage_datasets.py verify-saeczi
  python data/zero_inflated/scripts/manage_datasets.py zidf-synthetic --samples 10000 --zero-ratio 0.7
  python data/zero_inflated/scripts/manage_datasets.py zidf-noaa --start-year 1948 --end-year 1948
  pip install gdown && python data/zero_inflated/scripts/manage_datasets.py demm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
GDRIVE_ID = "1vy9h3uiarpwrFCGFgr9Ex81q2_1v0f34"
GITHUB_SHA = {
    "samp.rda": "6df7eb79d82f62777e2fe2f4fa02a18d7bf76bd3",
    "pop.rda": "e42b6a1ac2484069f0d2fadeca5ddd1b58cf74df",
}
NOAA = {
    "prate": ("surface_gauss", "prate.sfc.gauss"),
    "air": ("surface", "air.sig995"),
    "uwnd": ("surface", "uwnd.sig995"),
    "vwnd": ("surface", "vwnd.sig995"),
    "rhum": ("surface", "rhum.sig995"),
    "pres": ("surface", "pres.sfc"),
}
NOAA_ROOT = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs"


def git_blob_sha(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(b)).encode() + b"\x00" + b).hexdigest()


def verify_saeczi(_: argparse.Namespace) -> None:
    for name, expected in GITHUB_SHA.items():
        path = BASE / "saeczi" / "raw" / name
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        got = git_blob_sha(path)
        if got != expected:
            raise ValueError(f"SHA mismatch for {name}: got {got}, expected {expected}")
        print(f"[OK] {path.relative_to(BASE)}  ({path.stat().st_size} bytes)  Git SHA-1={got}")


def download_to(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[SKIP] exists: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.with_name(dst.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "research-datasets-organizer/1.0"})
    print(f"[GET] {url}")
    try:
        with urllib.request.urlopen(req, timeout=90) as src, temp.open("wb") as out:
            while True:
                block = src.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        if temp.stat().st_size == 0:
            raise ValueError("Downloaded empty file")
        temp.replace(dst)
        print(f"[OK] {dst} ({dst.stat().st_size} bytes)")
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def download_demm(_: argparse.Namespace) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install downloader first: pip install gdown") from exc
    output = BASE / "demm" / "raw" / "demm_author_bundle.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size:
        print("[SKIP] author bundle exists:", output)
        return
    print("Downloading author's ~400MB Google Drive bundle (not uploaded to Git).")
    result = gdown.download(id=GDRIVE_ID, output=str(output), quiet=False)
    if not result or not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Author's Google Drive download failed. Try the browser link in demm/README.md.")
    print("[OK] Downloaded", output)
    print("[NOTE] Inspect zip entries and extract manually; expected processed_data.pickle and rand_thresholds.pickle.")


def zidf_synthetic(args: argparse.Namespace) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Run: pip install numpy") from exc
    if args.features < 2:
        raise ValueError("--features must be at least 2")
    if args.samples <= 0 or not 0 <= args.zero_ratio <= 1:
        raise ValueError("Need samples > 0 and 0 <= zero-ratio <= 1")
    # Implements the method in upstream ZIDF data/preprocessing.py; RandomState mimics np.random.seed.
    rng = np.random.RandomState(args.seed)
    n = args.samples
    t = np.arange(n)
    periodic = np.sin(2 * np.pi * t / 365)[:, None]
    linear = (t / n)[:, None]
    feats = np.hstack([periodic, linear, rng.randn(n, args.features - 2)])
    y = rng.gamma(2.0, 0.05, n)
    y[rng.rand(n) < args.zero_ratio] = 0
    y *= 1 + 0.3 * (periodic[:, 0] + 1) / 2
    out = BASE / "zidf" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"synthetic_zero{round(args.zero_ratio * 100)}_n{n}_seed{args.seed}"
    path = out / (stem + ".npz")
    np.savez_compressed(path, features=feats, targets=y)
    meta = {
        "type": "SYNTHETIC_NOT_OBSERVED",
        "implementation": "ZIDF official generate_synthetic_data mathematical recipe",
        "n_samples": n,
        "n_features": args.features,
        "target_distribution": "Gamma(shape=2, scale=0.05) with Bernoulli zero mask",
        "seed": args.seed,
        "requested_zero_ratio": args.zero_ratio,
        "actual_zero_ratio": float(np.mean(y == 0)),
        "files": {"npz": path.name, "arrays": ["features", "targets"]},
    }
    (out / (stem + ".metadata.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] {path} features={feats.shape} targets={y.shape} zero={meta['actual_zero_ratio']:.4f}")


def zidf_noaa(args: argparse.Namespace) -> None:
    if not (1948 <= args.start_year <= args.end_year <= 2026):
        raise ValueError("Use a year range between 1948 and 2026.")
    variables = args.variables.split(",")
    if any(v not in NOAA for v in variables):
        raise ValueError("Allowed variables: " + ", ".join(NOAA))
    count = (args.end_year - args.start_year + 1) * len(variables)
    print(f"Preparing {count} NOAA annual NetCDF files; check disk space and provider terms.")
    for year in range(args.start_year, args.end_year + 1):
        for var in variables:
            folder, filename = NOAA[var]
            target = BASE / "zidf" / "raw" / str(year) / f"{filename}.{year}.nc"
            url = f"{NOAA_ROOT}/{folder}/{filename}.{year}.nc"
            if args.dry_run:
                print("[DRY RUN]", url, "->", target)
            else:
                download_to(url, target)
    print("These are NOAA *raw* files. Their paths and time/units must be preprocessed to match ZIDF configs.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demm", help="Download author's DEMM Google Drive ~400MB bundle")
    sub.add_parser("verify-saeczi", help="Verify two included original R datasets")
    syn = sub.add_parser("zidf-synthetic", help="Generate ZIDF-style zero-inflated Gamma data")
    syn.add_argument("--samples", type=int, default=10000)
    syn.add_argument("--features", type=int, default=5)
    syn.add_argument("--zero-ratio", type=float, default=0.7)
    syn.add_argument("--seed", type=int, default=42)
    noa = sub.add_parser("zidf-noaa", help="Fetch official NOAA daily NetCDF originals")
    noa.add_argument("--start-year", type=int, default=1948)
    noa.add_argument("--end-year", type=int, default=1948)
    noa.add_argument("--variables", default="prate,air,uwnd,vwnd,rhum,pres")
    noa.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        {"demm": download_demm, "verify-saeczi": verify_saeczi,
         "zidf-synthetic": zidf_synthetic, "zidf-noaa": zidf_noaa}[args.command](args)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
