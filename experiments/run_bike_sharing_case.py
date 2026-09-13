"""Timestamped Bike Sharing demand evidence-card case study.

Dataset: UCI Bike Sharing hourly demand data. Unlike the healthcare proxy case,
this dataset contains explicit calendar dates and hours. We use a chronological
split that mimics deployment:
- discovery: January--June 2011
- validation: July--December 2011
- held-out: all of 2012

The binary target is high hourly rental demand (cnt >= 200). Hypotheses are
written in the constrained natural-language grammar and verified by the same
unchanged verifier used elsewhere in this artifact.
"""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verifiable_discovery import Dataset, EvidenceCard, verify_hypotheses  # noqa: E402


DATA = ROOT / "data" / "real"
RESULTS = ROOT / "results"
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


HYPOTHESES = [
    "IF x0 = 1 THEN y = 1 increases",
    "IF x1 = 1 THEN y = 1 increases",
    "IF x2 = 1 THEN y = 1 increases",
    "IF x3 = 1 THEN y = 1 increases",
    "IF x4 = 1 THEN y = 1 increases",
    "IF x5 = 1 THEN y = 1 increases",
    "IF x6 = 1 THEN y = 1 decreases",
    "IF x7 = 1 THEN y = 1 decreases",
    "IF x8 = 1 THEN y = 1 increases",
    "IF x9 = 1 THEN y = 1 decreases",
    "IF x10 = 1 THEN y = 1 increases",
    "IF x11 = 1 THEN y = 1 decreases",
    "IF x12 = 1 THEN y = 1 decreases",
    "IF x13 = 1 THEN y = 1 decreases",
    "IF x14 = 1 THEN y = 1 increases",
    "IF x15 = 1 THEN y = 1 decreases",
    "IF x16 = 1 THEN y = 1 decreases",
    "IF x17 = 1 THEN y = 1 increases",
    "IF x18 = 1 THEN y = 1 increases",
    "IF x19 = 1 THEN y = 1 increases",
    "IF x20 = 1 THEN y = 1 increases",
    "IF x21 = 1 THEN y = 1 decreases",
    "IF x22 = 1 THEN y = 1 increases",
    "IF x1 = 1 AND x3 = 1 THEN y = 1 increases",
    "IF x1 = 1 AND x4 = 1 THEN y = 1 increases",
    "IF x7 = 1 AND x5 = 1 THEN y = 1 increases",
    "IF x8 = 1 AND x10 = 1 THEN y = 1 increases",
    "IF x9 = 1 AND x12 = 1 THEN y = 1 decreases",
    "IF x15 = 1 AND x6 = 1 THEN y = 1 decreases",
    "IF x14 = 1 AND x4 = 1 THEN y = 1 increases",
    "IF x18 = 1 AND x2 = 1 THEN y = 1 increases",
    "IF x13 = 1 AND x10 = 1 THEN y = 1 decreases",
    "IF x20 = 1 AND x8 = 1 THEN y = 1 increases",
    "IF x21 = 1 AND x6 = 1 THEN y = 1 decreases",
    "IF x22 = 1 AND x4 = 1 THEN y = 1 increases",
    "IF SEQUENCE x1 -> x4 WITHIN 24 HOURS THEN y = 1 increases",
]


FEATURE_FAMILIES = {
    "bike_warm_season": "season",
    "bike_workingday": "calendar",
    "bike_commute_hour": "hour",
    "bike_morning_peak": "hour",
    "bike_evening_peak": "hour",
    "bike_midday": "hour",
    "bike_night": "hour",
    "bike_weekend": "calendar",
    "bike_clear_weather": "weather",
    "bike_bad_weather": "weather",
    "bike_temp_high": "weather",
    "bike_temp_low": "weather",
    "bike_humidity_high": "weather",
    "bike_wind_high": "weather",
    "bike_summer": "season",
    "bike_winter": "season",
    "bike_holiday": "calendar",
    "bike_weekday": "calendar",
    "bike_fall": "season",
    "bike_spring": "season",
    "bike_temp_mild": "weather",
    "bike_cold_night": "weather+hour",
    "bike_september": "calendar",
    "seq_x1_x4_within_24d": "sequence",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fetch_bike() -> pd.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    cached = DATA / "bike_sharing_hour.csv"
    if cached.exists():
        return pd.read_csv(cached)
    with urlopen(URL, timeout=120) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open("hour.csv") as f:
            frame = pd.read_csv(f)
    frame.to_csv(cached, index=False)
    return frame


def load_dataset() -> tuple[Dataset, pd.DataFrame]:
    frame = fetch_bike().copy()
    frame["date"] = pd.to_datetime(frame["dteday"])
    features = pd.DataFrame(
        {
            "bike_warm_season": frame["season"].isin([2, 3]),
            "bike_workingday": frame["workingday"].eq(1),
            "bike_commute_hour": frame["hr"].between(7, 9) | frame["hr"].between(16, 19),
            "bike_morning_peak": frame["hr"].between(7, 9),
            "bike_evening_peak": frame["hr"].between(16, 19),
            "bike_midday": frame["hr"].between(11, 15),
            "bike_night": frame["hr"].between(0, 5),
            "bike_weekend": frame["weekday"].isin([0, 6]),
            "bike_clear_weather": frame["weathersit"].eq(1),
            "bike_bad_weather": frame["weathersit"].ge(3),
            "bike_temp_high": frame["temp"].ge(0.60),
            "bike_temp_low": frame["temp"].le(0.30),
            "bike_humidity_high": frame["hum"].ge(0.75),
            "bike_wind_high": frame["windspeed"].ge(0.30),
            "bike_summer": frame["season"].eq(3),
            "bike_winter": frame["season"].eq(1),
            "bike_holiday": frame["holiday"].eq(1),
            "bike_weekday": ~frame["weekday"].isin([0, 6]),
            "bike_fall": frame["season"].eq(4),
            "bike_spring": frame["season"].eq(2),
            "bike_temp_mild": frame["temp"].between(0.40, 0.70),
            "bike_cold_night": frame["temp"].le(0.30) & frame["hr"].between(0, 5),
            "bike_september": frame["mnth"].eq(9),
            # Real timestamped sequence feature: a working-day context followed
            # by the evening commute window within the same 24-hour calendar day.
            "seq_x1_x4_within_24d": frame["workingday"].eq(1) & frame["hr"].between(16, 19),
        }
    )
    split = np.full(len(frame), -1, dtype=int)
    split[(frame["yr"].eq(0)) & (frame["mnth"].le(6))] = 0
    split[(frame["yr"].eq(0)) & (frame["mnth"].ge(7))] = 1
    split[frame["yr"].eq(1)] = 2
    keep = split >= 0
    target = frame["cnt"].ge(200).astype(int)
    data = Dataset(
        x=features.loc[keep].astype(int).to_numpy(),
        y=target.loc[keep].to_numpy(),
        split=split[keep],
        feature_names=tuple(features.columns),
    )
    return data, frame.loc[keep].reset_index(drop=True)


def case_summary(frame: pd.DataFrame, data: Dataset, cards: list[EvidenceCard]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    names = {
        0: "discovery_2011H1",
        1: "validation_2011H2",
        2: "holdout_2012",
    }
    for split_id, name in names.items():
        mask = data.split == split_id
        rows.append(
            {
                "split": name,
                "n": int(mask.sum()),
                "high_demand_rate": float(data.y[mask].mean()),
                "start_date": str(frame.loc[mask, "date"].min().date()),
                "end_date": str(frame.loc[mask, "date"].max().date()),
                "hypotheses": len(HYPOTHESES),
                "accepted": sum(card.decision == "accepted" for card in cards),
                "rejected": sum(card.decision == "rejected" for card in cards),
            }
        )
    return rows


def evidence_rows(cards: list[EvidenceCard]) -> list[dict[str, object]]:
    rows = []
    for card in cards:
        antecedent = card.query.split(" -> ")[0]
        families = sorted(
            {
                FEATURE_FAMILIES.get(part.strip().split(" ")[0], "other")
                for part in antecedent.split(" AND ")
            }
        )
        rows.append(
            {
                "dataset": "BikeSharingHourly",
                "split": "2011H1/2011H2/2012",
                "decision": card.decision,
                "hypothesis": card.hypothesis,
                "query": card.query,
                "family": "+".join(families),
                "support_validation": card.support_validation,
                "support_holdout": card.support_holdout,
                "validation_effect": card.validation_effect,
                "p_value": card.p_value,
                "q_value": card.q_value,
                "counterfactual_effect": card.counterfactual_effect,
                "holdout_effect": card.holdout_effect,
                "holdout_ci_low": card.holdout_ci[0],
                "holdout_ci_high": card.holdout_ci[1],
                "reason": card.reason,
            }
        )
    return rows


def main() -> None:
    data, frame = load_dataset()
    cards = verify_hypotheses(
        data,
        HYPOTHESES,
        permutation_rounds=500,
        bootstrap_rounds=500,
        min_support=0.025,
        min_effect=0.05,
        alpha=0.10,
        use_support=True,
        use_statistical_test=True,
        use_fdr=True,
        use_counterfactual=True,
        use_holdout=True,
    )
    rows = evidence_rows(cards)
    rows.sort(key=lambda row: (row["decision"] != "accepted", -abs(float(row["holdout_effect"]))))
    write_csv(RESULTS / "bike_sharing_evidence_cards.csv", rows)
    write_csv(RESULTS / "bike_sharing_case_summary.csv", case_summary(frame, data, cards))
    accepted = sum(card.decision == "accepted" for card in cards)
    rejected = sum(card.decision == "rejected" for card in cards)
    print(f"Bike Sharing case: {accepted} accepted, {rejected} rejected, {len(cards)} total")


if __name__ == "__main__":
    main()
