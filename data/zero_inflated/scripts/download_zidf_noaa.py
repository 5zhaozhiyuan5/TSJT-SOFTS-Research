#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download real NOAA NCEP/NCAR daily NetCDF files for ZIDF's South Australia study.

The NOAA NetCDF files are REAL reanalysis data, not ZIDF synthetic Gamma data.
This is an independent preparation pipeline and does not guarantee identical
preprocessing to the published paper.

Examples from the repository root:
    python data/zero_inflated/scripts/download_zidf_noaa.py download --start-year 1948 --end-year 1948
    python data/zero_inflated/scripts/download_zidf_noaa.py download --start-year 1948 --end-year 2014
    python data/zero_inflated/scripts/download_zidf_noaa.py verify
    pip install xarray netCDF4 pandas numpy
    python data/zero_inflated/scripts/download_zidf_noaa.py prepare

If NOAA's subsetting endpoint is unavailable, download complete global
annual files with --mode global. Default mode is bbox subset to save bandwidth.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = Path(__file__).resolve().parents[1] / "zidf"
NOAA = "https://psl.noaa.gov/thredds"
VARS = {
    "prate": ("surface_gauss", "prate.sfc.gauss"),
    "air": ("surface", "air.sig995"),
    "uwnd": ("surface", "uwnd.sig995"),
    "vwnd": ("surface", "vwnd.sig995"),
    "rhum": ("surface", "rhum.sig995"),
    "pres": ("surface_gauss", "pres.sfc.gauss"),
}


def is_netcdf(p):
    try:
        if p.stat().st_size < 1024:
            return False
        with p.open("rb") as f:
            head = f.read(8)
        return head.startswith((b"CDF\x01", b"CDF\x02", b"\x89HDF\r\n\x1a\n"))
    except OSError:
        return False


def url_for(var, year, args):
    directory, stem = VARS[var]
    resource = f"Datasets/ncep.reanalysis.dailyavgs/{directory}/{stem}.{year}.nc"
    if args.mode == "global":
        return f"{NOAA}/fileServer/{resource}"
    query = urlencode({
        "var": var, "north": args.north, "south": args.south,
        "west": args.west, "east": args.east, "time": "all",
        "horizStride": 1, "addLatLon": "true", "accept": "netcdf4"
    })
    return f"{NOAA}/ncss/grid/{resource}?{query}"


def path_for(var, year, args):
    return args.output / args.mode / str(year) / f"{VARS[var][1]}.{year}.nc"


def fetch(url, path, args):
    if is_netcdf(path):
        return "skipped_validated", path.stat().st_size
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    last = None
    for attempt in range(1, args.retries + 1):
        try:
            request = Request(
                url, headers={"User-Agent": "Academic-NOAA-ZIDF-Dataset-Downloader/1.0"}
            )
            with urlopen(request, timeout=args.timeout) as res, tmp.open("wb") as out:
                if res.status != 200:
                    raise RuntimeError(f"HTTP {res.status}")
                while block := res.read(1024 * 1024):
                    out.write(block)
            if not is_netcdf(tmp):
                with tmp.open("rb") as f:
                    head = f.read(160)
                raise RuntimeError(f"not NetCDF; received {tmp.stat().st_size} bytes, first bytes {head!r}")
            tmp.replace(path)
            return "downloaded_validated", path.stat().st_size
        except (HTTPError, URLError, OSError, TimeoutError, RuntimeError) as exc:
            last = exc
            tmp.unlink(missing_ok=True)
            if attempt != args.retries:
                pause = min(2 ** attempt * 2, 35)
                print(f"  Retry {attempt}/{args.retries}: {exc} (sleep {pause}s)")
                time.sleep(pause)
    raise RuntimeError(f"Failed: {url} ({last})")


def download(args):
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / f"download_manifest_{args.mode}.jsonl"
    failures = 0
    successes = 0
    for year in range(args.start_year, args.end_year + 1):
        for var in args.vars:
            link = url_for(var, year, args)
            path = path_for(var, year, args)
            print(f"[{year} {var}] {path}")
            if args.dry_run:
                print("  URL:", link)
                continue
            record = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "year": year, "variable": var, "url": link, "path": str(path)
            }
            try:
                record["status"], record["bytes"] = fetch(link, path, args)
                successes += 1
                print(f"  OK ({record['status']}: {record['bytes']:,} bytes)")
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
                failures += 1
                print(f"  FAILED: {exc}", file=sys.stderr)
            with manifest.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.delay:
                time.sleep(args.delay)
    print(f"Finished. {successes} verified, {failures} failed; manifest: {manifest}")
    if failures:
        raise RuntimeError("Download incomplete. Rerun exactly the same command to skip valid files and retry failed files.")


def verify(args):
    ok, missing = 0, []
    for year in range(args.start_year, args.end_year + 1):
        for var in args.vars:
            path = path_for(var, year, args)
            if is_netcdf(path):
                ok += 1
            else:
                missing.append(str(path))
    print(f"NetCDF headers verified: {ok}/{ok + len(missing)}")
    if missing:
        print("Missing files:", *missing[:20], sep="\n - ")
        raise RuntimeError(f"{len(missing)} missing / incomplete files")


def prepare(args):
    try:
        import numpy as np
        import pandas as pd
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("Install the processing libraries: pip install numpy pandas xarray netCDF4") from exc
    if "prate" not in args.vars:
        raise ValueError("prate is required for precipitation prediction")
    frames = []
    for year in range(args.start_year, args.end_year + 1):
        series = []
        for var in args.vars:
            path = path_for(var, year, args)
            if not is_netcdf(path):
                raise FileNotFoundError(path)
            with xr.open_dataset(path) as ds:
                if var not in ds:
                    raise KeyError(f"{var} missing in {path}: {list(ds.data_vars)}")
                data = ds[var]
                if args.mode == "global":
                    lat_asc = bool(data.lat.values[0] < data.lat.values[-1])
                    lats = slice(args.south, args.north) if lat_asc else slice(args.north, args.south)
                    data = data.sel(lat=lats, lon=slice(args.west, args.east))
                if not all(d in data.dims and data.sizes[d] > 0 for d in ("lat", "lon")):
                    raise ValueError(f"No spatial data after South Australia selection: {path}")
                units = str(data.attrs.get("units", "")).lower().replace(" ", "")
                data = data.mean(dim=("lat", "lon"), skipna=True)
                if var == "prate":
                    if "kg" in units and any(u in units for u in ("/s", "s-1", "s^-1")):
                        data *= 86400.0    # kg/m^2/s -> mm/day
                    else:
                        raise ValueError(f"Unverified precipitation units for {path}: {units}")
                values = data.to_series()
                values.index = pd.to_datetime(values.index).normalize()
                values = values.groupby(level=0).mean()
                values.name = var
                series.append(values)
        frame = pd.concat(series, axis=1, join="inner").sort_index()
        if len(frame) < 300 or frame.isna().any().any():
            raise ValueError(f"Insufficient or missing aligned observations in {year}: {frame.shape}")
        frames.append(frame)
        print(year, "days:", len(frame))
    all_years = pd.concat(frames).sort_index()
    if all_years.index.duplicated().any():
        raise ValueError("Duplicate dates")
    if (all_years.prate < -1e-8).any():
        raise ValueError("Negative precipitation unexpectedly found")
    all_years.prate = all_years.prate.clip(lower=0)
    target = all_years.prate.to_numpy(dtype="float32")
    features = all_years[[v for v in args.vars if v != "prate"]].to_numpy(dtype="float32")
    output = BASE / "processed"
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / f"sa_{args.start_year}_{args.end_year}_{args.mode}_regionalmean"
    all_years.rename(columns={"prate": "precip_mm_per_day"}).to_csv(
        str(prefix) + ".csv", index_label="date"
    )
    np.savez_compressed(str(prefix) + ".npz", features=features, targets=target,
                        dates=all_years.index.strftime("%Y-%m-%d").to_numpy())
    details = {
        "source": "NOAA NCEP/NCAR Reanalysis 1", "source_type": "REAL_REANALYSIS",
        "processing": "Area mean; custom pipeline, not verified as exact original-paper preprocessing",
        "years": [args.start_year, args.end_year],
        "region": [args.south, args.north, args.west, args.east],
        "rows": len(all_years), "zero_ratio": float(np.mean(target == 0)),
        "target_units": "mm/day", "features": [v for v in args.vars if v != "prate"]
    }
    Path(str(prefix) + ".json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print("Saved:", str(prefix), ".csv / .npz / .json")
    print("Observed zero fraction:", details["zero_ratio"])
    print("WARNING: Spatial averaging can greatly change the exact-zero ratio compared with the paper.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "verify", "prepare"):
        p = sub.add_parser(command)
        p.add_argument("--start-year", type=int, default=1948)
        p.add_argument("--end-year", type=int, default=2014)
        p.add_argument("--mode", choices=["subset", "global"], default="subset")
        p.add_argument("--vars", default=",".join(VARS))
        p.add_argument("--south", type=float, default=-38.0)
        p.add_argument("--north", type=float, default=-26.0)
        p.add_argument("--west", type=float, default=129.0)
        p.add_argument("--east", type=float, default=141.0)
        p.add_argument("--output", type=Path, default=BASE / "raw" / "noaa_sa")
        if command == "download":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--retries", type=int, default=4)
            p.add_argument("--timeout", type=int, default=150)
            p.add_argument("--delay", type=float, default=0.4)
    args = parser.parse_args()
    if args.start_year < 1948 or args.end_year > 2026 or args.start_year > args.end_year:
        parser.error("Year range invalid; allowed 1948..2026")
    if not (-90 <= args.south < args.north <= 90 and 0 <= args.west < args.east <= 360):
        parser.error("Geographic bounds invalid")
    args.vars = [v.strip() for v in args.vars.split(",") if v.strip()]
    if not args.vars or len(set(args.vars)) != len(args.vars) or any(v not in VARS for v in args.vars):
        parser.error("Variables must be a unique subset of: " + ", ".join(VARS))
    try:
        {"download": download, "verify": verify, "prepare": prepare}[args.command](args)
    except Exception as exc:
        print("[ERROR]", exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
