"""Analyze VeraDM human audit study responses.

By default this analyzes the synthetic demo response file. For real results,
pass a CSV with the same columns as results/human_audit_response_template.csv
after participants have filled response_action, confidence_1_to_5, and
time_seconds.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(path: Path) -> list[dict[str, object]]:
    responses = pd.read_csv(path)
    key = pd.read_csv(RESULTS / "human_audit_answer_key.csv")
    frame = responses.merge(key, on="item_id", how="left")
    frame = frame[frame["response_action"].notna() & (frame["response_action"].astype(str).str.len() > 0)].copy()
    if len(frame) == 0:
        raise ValueError("no completed responses found")
    frame["correct"] = frame["response_action"].eq(frame["correct_action"])
    frame["confidence_1_to_5"] = pd.to_numeric(frame["confidence_1_to_5"], errors="coerce")
    frame["time_seconds"] = pd.to_numeric(frame["time_seconds"], errors="coerce")

    rows: list[dict[str, object]] = []
    for condition, group in frame.groupby("condition", sort=True):
        rows.append(
            {
                "condition": condition,
                "responses": len(group),
                "participants": group["participant_id"].nunique(),
                "accuracy": group["correct"].mean(),
                "mean_confidence": group["confidence_1_to_5"].mean(),
                "median_time_seconds": group["time_seconds"].median(),
            }
        )
    if set(frame["condition"]) >= {"evidence_card", "prose_only"}:
        card = frame[frame["condition"] == "evidence_card"]["correct"].mean()
        prose = frame[frame["condition"] == "prose_only"]["correct"].mean()
        rows.append(
            {
                "condition": "difference_card_minus_prose",
                "responses": len(frame),
                "participants": frame["participant_id"].nunique(),
                "accuracy": card - prose,
                "mean_confidence": "",
                "median_time_seconds": "",
            }
        )
    return rows


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "human_audit_demo_responses.csv"
    rows = analyze(path)
    out = RESULTS / ("human_audit_demo_summary.csv" if path.name == "human_audit_demo_responses.csv" else "human_audit_summary.csv")
    write_csv(out, rows)
    print(f"Wrote {out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
