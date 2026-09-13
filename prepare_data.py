#!/usr/bin/env python3
"""Fetch and cache the public datasets the experiments need.

Bike Sharing and Diabetes readmission ship pre-cached in data/real/. Adult and
German Credit cache themselves on first use. Bank Marketing needs one derived
column pair, which this script builds from the public UCI release.

    python prepare_data.py --bank      # build data/real/bank_month_env.csv
    python prepare_data.py --all       # the above, plus Adult and German
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "real"
BANK_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

# Row counts of the three calendar splits, used as a self-check.
EXPECTED = {"discovery_may_jun": 19087, "validation_jul_aug": 13352, "holdout_nov": 4101}


def build_bank() -> Path:
    """bank-additional-full.csv + env(month) + target(y) -> bank_month_env.csv."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "bank_month_env.csv"
    if out.exists():
        print(f"already present: {out}")
        return out
    print(f"fetching {BANK_URL}")
    with urlopen(BANK_URL, timeout=180) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        inner_name = next(n for n in outer.namelist() if n.endswith("bank-additional.zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
            csv_name = next(n for n in inner.namelist() if n.endswith("bank-additional-full.csv"))
            frame = pd.read_csv(io.BytesIO(inner.read(csv_name)), sep=";")

    frame["env"] = frame["month"]
    frame["target"] = frame["y"].eq("yes").astype(int)

    got = {
        "discovery_may_jun": int(frame["env"].isin(["may", "jun"]).sum()),
        "validation_jul_aug": int(frame["env"].isin(["jul", "aug"]).sum()),
        "holdout_nov": int(frame["env"].eq("nov").sum()),
    }
    if got != EXPECTED:
        sys.exit(f"split sizes {got} do not match expected {EXPECTED}; refusing to write {out}")

    frame.to_csv(out, index=False)
    print(f"wrote {out}  ({len(frame)} rows; splits {got})")
    return out


def build_adult_and_german() -> None:
    sys.path.insert(0, str(ROOT / "experiments"))
    sys.path.insert(0, str(ROOT / "src"))
    from run_verifiable_discovery import load_adult_income, load_german_credit

    load_adult_income()
    load_german_credit()
    print(f"cached: {DATA / 'adult_income.csv'}, {DATA / 'german_credit.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", action="store_true", help="build bank_month_env.csv")
    ap.add_argument("--all", action="store_true", help="bank, plus Adult and German")
    args = ap.parse_args()
    if not (args.bank or args.all):
        ap.error("choose --bank or --all")
    build_bank()
    if args.all:
        build_adult_and_german()


if __name__ == "__main__":
    main()
