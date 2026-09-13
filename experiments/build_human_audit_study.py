"""Build a human audit study packet for VeraDM evidence cards.

This script creates the materials needed to run a real within-subjects audit:
- blinded task items with prose-only and evidence-card conditions
- an answer key based on verifier decisions
- a response-template CSV for participants

Real responses are collected by filling human_audit_response_template.csv
and analyzed with experiments/analyze_human_audit.py.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"


SOURCE_FILES = [
    ("Bank", RESULTS / "bank_temporal_evidence_cards.csv"),
    ("Bike", RESULTS / "bike_sharing_evidence_cards.csv"),
    ("Healthcare", RESULTS / "healthcare_readmission_evidence_cards.csv"),
]


SELECTED_QUERIES = {
    "Bank": [
        "bank_duration_ge_300 = 1 AND bank_euribor_low = 1 -> P(y=1) increases",
        "bank_job_blue_collar = 1 AND bank_campaign_gt_2 = 1 -> P(y=1) decreases",
        "bank_poutcome_success = 1 -> P(y=1) increases",
        "bank_contact_cellular = 1 -> P(y=1) increases",
    ],
    "Bike": [
        "bike_workingday = 1 AND bike_evening_peak = 1 -> P(y=1) increases",
        "bike_bad_weather = 1 AND bike_humidity_high = 1 -> P(y=1) decreases",
        "bike_winter = 1 AND bike_night = 1 -> P(y=1) decreases",
        "bike_wind_high = 1 AND bike_temp_high = 1 -> P(y=1) decreases",
    ],
    "Healthcare": [
        "diab_inpatient_prior = 1 -> P(y=1) increases",
        "diab_no_prior_utilization = 1 -> P(y=1) decreases",
        "diab_admission_emergency = 1 -> P(y=1) increases",
        "diab_num_lab_procedures_ge_60 = 1 -> P(y=1) increases",
    ],
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def short_rule(query: str) -> str:
    return query.replace(" = 1", "").replace(" -> P(y=1) ", " -> outcome ")


def prose_summary(dataset: str, row: pd.Series) -> str:
    direction = "higher" if "increases" in row["query"] else "lower"
    return (
        f"In the {dataset} data, the rule '{short_rule(row['query'])}' appears to be associated with "
        f"{direction} target probability in the validation data. Would you act on this rule?"
    )


def evidence_card(row: pd.Series) -> str:
    return (
        f"Claim: {row['hypothesis']} | Query: {row['query']} | "
        f"Validation support/effect: {row['support_validation']:.3f}/{row['validation_effect']:.3f} | "
        f"q: {row['q_value']:.3f} | Matched contrast: {row['counterfactual_effect']:.3f} | "
        f"Held-out support/effect/CI: {row['support_holdout']:.3f}/{row['holdout_effect']:.3f}/"
        f"[{row['holdout_ci_low']:.3f},{row['holdout_ci_high']:.3f}] | "
        f"Verifier decision: {row['decision']} | Reason: {row['reason']}"
    )


def build_items() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, path in SOURCE_FILES:
        frame = pd.read_csv(path)
        for query in SELECTED_QUERIES[dataset]:
            match = frame[frame["query"] == query]
            if len(match) != 1:
                raise ValueError(f"expected one row for {dataset}: {query}")
            row = match.iloc[0]
            rows.append(
                {
                    "item_id": f"{dataset.lower()}_{len(rows) + 1:02d}",
                    "dataset": dataset,
                    "query": row["query"],
                    "family": row["family"],
                    "verifier_decision": row["decision"],
                    "correct_action": "act" if row["decision"] == "accepted" else "do_not_act",
                    "prose_only_prompt": prose_summary(dataset, row),
                    "evidence_card_prompt": evidence_card(row),
                    "validation_effect": f"{row['validation_effect']:.6f}",
                    "holdout_effect": f"{row['holdout_effect']:.6f}",
                    "q_value": f"{row['q_value']:.6f}",
                    "reason": row["reason"],
                }
            )
    return rows


def build_assignment(items: list[dict[str, object]], participants: int = 16) -> list[dict[str, object]]:
    rng = random.Random(17)
    assignments: list[dict[str, object]] = []
    for participant in range(1, participants + 1):
        shuffled = items[:]
        rng.shuffle(shuffled)
        for order, item in enumerate(shuffled, start=1):
            condition = "evidence_card" if (participant + order) % 2 == 0 else "prose_only"
            prompt_key = "evidence_card_prompt" if condition == "evidence_card" else "prose_only_prompt"
            assignments.append(
                {
                    "participant_id": f"P{participant:02d}",
                    "order": order,
                    "item_id": item["item_id"],
                    "condition": condition,
                    "prompt": item[prompt_key],
                    "response_action": "",
                    "confidence_1_to_5": "",
                    "time_seconds": "",
                    "notes": "",
                }
            )
    return assignments


def write_protocol() -> None:
    protocol = """# VeraDM Human Audit Study Protocol

Goal: test whether evidence cards help analysts decide which candidate discoveries are trustworthy.

Design: within-subjects randomized audit. Each participant sees 12 candidate rules from Bank Marketing, Bike Sharing, and Healthcare. Half are shown as prose-only summaries and half as evidence cards. Assignment order and condition are counterbalanced.

Participants: target n=16-20 graduate students, data analysts, or domain-adjacent researchers. Exclude authors of the study.

Task: for each item, choose `act` or `do_not_act`, report confidence from 1 to 5, and optionally leave notes.

Ground truth for scoring: verifier decision. Accepted cards map to `act`; rejected cards map to `do_not_act`.

Primary metrics: decision accuracy by condition, mean confidence by condition, and median time by condition.

Secondary analysis: McNemar-style paired comparison or participant-clustered bootstrap for evidence-card vs prose-only accuracy.

Files:
- `results/human_audit_items.csv`: item bank and answer key fields.
- `results/human_audit_response_template.csv`: randomized participant worksheet with real responses.
- `experiments/analyze_human_audit.py`: analysis script for real responses.
"""
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "human_audit_study_protocol.md").write_text(protocol)


def main() -> None:
    items = build_items()
    key = {item["item_id"]: item["correct_action"] for item in items}
    assignments = build_assignment(items)
    write_csv(RESULTS / "human_audit_items.csv", items)
    write_csv(RESULTS / "human_audit_answer_key.csv", [{"item_id": k, "correct_action": v} for k, v in key.items()])
    write_csv(RESULTS / "human_audit_response_template.csv", assignments)
    write_protocol()
    print(f"Wrote {len(items)} audit items and {len(assignments)} randomized assignments")


if __name__ == "__main__":
    main()
