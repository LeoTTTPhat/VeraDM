"""Run a temporal Bank Marketing evidence-card case study.

This script uses the prepared Bank Marketing month-environment dataset and a
natural chronological split:
- discovery: May and June
- validation: July and August
- held-out: November

The hypotheses are written in the same constrained natural-language grammar used
by the LLM proposer. The verifier is unchanged.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verifiable_discovery import Dataset, EvidenceCard, verify_hypotheses  # noqa: E402


DATA = ROOT / "data" / "real" / "bank_month_env.csv"
RESULTS = ROOT / "results"


TEMPORAL_HYPOTHESES = [
    "IF x0 = 1 THEN y = 1 increases",
    "IF x1 = 1 THEN y = 1 decreases",
    "IF x2 = 1 THEN y = 1 increases",
    "IF x3 = 1 THEN y = 1 decreases",
    "IF x4 = 1 THEN y = 1 increases",
    "IF x5 = 1 THEN y = 1 increases",
    "IF x6 = 1 THEN y = 1 increases",
    "IF x7 = 1 THEN y = 1 decreases",
    "IF x8 = 1 THEN y = 1 decreases",
    "IF x9 = 1 THEN y = 1 decreases",
    "IF x10 = 1 THEN y = 1 increases",
    "IF x11 = 1 THEN y = 1 decreases",
    "IF x12 = 1 THEN y = 1 increases",
    "IF x13 = 1 THEN y = 1 increases",
    "IF x14 = 1 THEN y = 1 increases",
    "IF x15 = 1 THEN y = 1 decreases",
    "IF x16 = 1 THEN y = 1 increases",
    "IF x17 = 1 THEN y = 1 increases",
    "IF x18 = 1 THEN y = 1 increases",
    "IF x19 = 1 THEN y = 1 increases",
    "IF x20 = 1 THEN y = 1 increases",
    "IF x21 = 1 THEN y = 1 increases",
    "IF x22 = 1 THEN y = 1 decreases",
    "IF x23 = 1 THEN y = 1 increases",
    "IF x24 = 1 THEN y = 1 increases",
    "IF x25 = 1 THEN y = 1 increases",
    "IF x26 = 1 THEN y = 1 decreases",
    "IF x27 = 1 THEN y = 1 increases",
    "IF x13 = 1 AND x21 = 1 THEN y = 1 increases",
    "IF x17 = 1 AND x21 = 1 THEN y = 1 increases",
    "IF x18 = 1 AND x20 = 1 THEN y = 1 increases",
    "IF x4 = 1 AND x13 = 1 THEN y = 1 increases",
    "IF x10 = 1 AND x13 = 1 THEN y = 1 increases",
    "IF x15 = 1 AND x7 = 1 THEN y = 1 decreases",
    "IF x15 = 1 AND x9 = 1 THEN y = 1 decreases",
    "IF x3 = 1 AND x15 = 1 THEN y = 1 decreases",
    "IF x2 = 1 AND x6 = 1 THEN y = 1 increases",
    "IF x5 = 1 AND x10 = 1 THEN y = 1 increases",
    "IF x0 = 1 AND x4 = 1 THEN y = 1 increases",
]


FEATURE_FAMILIES = {
    "bank_age_ge_50": "demographic",
    "bank_married": "demographic",
    "bank_job_admin": "occupation",
    "bank_job_blue_collar": "occupation",
    "bank_job_retired": "occupation",
    "bank_job_student": "occupation",
    "bank_university_degree": "education",
    "bank_default_unknown": "credit",
    "bank_housing_yes": "credit",
    "bank_loan_yes": "credit",
    "bank_contact_cellular": "channel",
    "bank_contact_telephone": "channel",
    "bank_duration_ge_180": "interaction",
    "bank_duration_ge_300": "interaction",
    "bank_duration_ge_600": "interaction",
    "bank_campaign_gt_2": "campaign",
    "bank_campaign_eq_1": "campaign",
    "bank_previous_contact": "history",
    "bank_poutcome_success": "history",
    "bank_poutcome_failure": "history",
    "bank_pdays_recent": "history",
    "bank_euribor_low": "macro",
    "bank_euribor_high": "macro",
    "bank_emp_var_negative": "macro",
    "bank_nr_employed_low": "macro",
    "bank_confidence_ge_neg40": "macro",
    "bank_day_monday": "calendar",
    "bank_day_friday": "calendar",
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


def load_bank_temporal() -> Dataset:
    if not DATA.exists():
        raise FileNotFoundError(
            f"Bank Marketing data not found at {DATA}.\n"
            "Run: python prepare_data.py --bank"
        )
    frame = pd.read_csv(DATA)
    features = pd.DataFrame(
        {
            "bank_age_ge_50": frame["age"] >= 50,
            "bank_married": frame["marital"].eq("married"),
            "bank_job_admin": frame["job"].eq("admin."),
            "bank_job_blue_collar": frame["job"].eq("blue-collar"),
            "bank_job_retired": frame["job"].eq("retired"),
            "bank_job_student": frame["job"].eq("student"),
            "bank_university_degree": frame["education"].eq("university.degree"),
            "bank_default_unknown": frame["default"].eq("unknown"),
            "bank_housing_yes": frame["housing"].eq("yes"),
            "bank_loan_yes": frame["loan"].eq("yes"),
            "bank_contact_cellular": frame["contact"].eq("cellular"),
            "bank_contact_telephone": frame["contact"].eq("telephone"),
            "bank_duration_ge_180": frame["duration"] >= 180,
            "bank_duration_ge_300": frame["duration"] >= 300,
            "bank_duration_ge_600": frame["duration"] >= 600,
            "bank_campaign_gt_2": frame["campaign"] > 2,
            "bank_campaign_eq_1": frame["campaign"] == 1,
            "bank_previous_contact": frame["previous"] > 0,
            "bank_poutcome_success": frame["poutcome"].eq("success"),
            "bank_poutcome_failure": frame["poutcome"].eq("failure"),
            "bank_pdays_recent": frame["pdays"] < 999,
            "bank_euribor_low": frame["euribor3m"] < 2.0,
            "bank_euribor_high": frame["euribor3m"] > 4.8,
            "bank_emp_var_negative": frame["emp.var.rate"] < 0.0,
            "bank_nr_employed_low": frame["nr.employed"] < 5100.0,
            "bank_confidence_ge_neg40": frame["cons.conf.idx"] >= -40.0,
            "bank_day_monday": frame["day_of_week"].eq("mon"),
            "bank_day_friday": frame["day_of_week"].eq("fri"),
        }
    )
    split = np.full(len(frame), -1, dtype=int)
    split[frame["env"].isin(["may", "jun"])] = 0
    split[frame["env"].isin(["jul", "aug"])] = 1
    split[frame["env"].isin(["nov"])] = 2
    keep = split >= 0
    return Dataset(
        x=features.loc[keep].astype(int).to_numpy(),
        y=frame.loc[keep, "target"].astype(int).to_numpy(),
        split=split[keep],
        feature_names=tuple(features.columns),
    )


def case_summary(frame: pd.DataFrame, cards: list[EvidenceCard]) -> list[dict[str, object]]:
    split_names = {0: "discovery_MayJun", 1: "validation_JulAug", 2: "holdout_Nov"}
    rows: list[dict[str, object]] = []
    for split_id, split_name in split_names.items():
        mask = frame["split"] == split_id
        rows.append(
            {
                "metric": split_name,
                "n": int(mask.sum()),
                "target_rate": float(frame.loc[mask, "target"].mean()),
                "hypotheses": len(TEMPORAL_HYPOTHESES),
                "accepted": sum(card.decision == "accepted" for card in cards),
                "rejected": sum(card.decision == "rejected" for card in cards),
            }
        )

    reason_counts = {
        "insufficient_support": 0,
        "weak_validation": 0,
        "not_significant": 0,
        "weak_contrast": 0,
        "heldout_failure": 0,
    }
    for card in cards:
        reason_counts["insufficient_support"] += int("insufficient support" in card.reason)
        reason_counts["weak_validation"] += int("weak validation effect" in card.reason)
        reason_counts["not_significant"] += int("not significant" in card.reason)
        reason_counts["weak_contrast"] += int("matched counterfactual" in card.reason)
        reason_counts["heldout_failure"] += int("held-out effect" in card.reason)
    for key, value in reason_counts.items():
        rows.append(
            {
                "metric": key,
                "n": value,
                "target_rate": "",
                "hypotheses": len(TEMPORAL_HYPOTHESES),
                "accepted": "",
                "rejected": "",
            }
        )
    return rows


def evidence_rows(cards: list[EvidenceCard]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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
                "dataset": "BankMarketingMonth",
                "split": "May-Jun/Jul-Aug/Nov",
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
    data = load_bank_temporal()
    target_frame = pd.DataFrame({"split": data.split, "target": data.y})
    cards = verify_hypotheses(
        data,
        TEMPORAL_HYPOTHESES,
        permutation_rounds=500,
        bootstrap_rounds=500,
        min_support=0.025,
        min_effect=0.035,
        alpha=0.10,
        use_support=True,
        use_statistical_test=True,
        use_fdr=True,
        use_counterfactual=True,
        use_holdout=True,
    )
    rows = evidence_rows(cards)
    write_csv(RESULTS / "bank_temporal_evidence_cards.csv", rows)
    write_csv(RESULTS / "bank_temporal_case_summary.csv", case_summary(target_frame, cards))

    accepted = [row for row in rows if row["decision"] == "accepted"]
    rejected = [row for row in rows if row["decision"] == "rejected"]
    print("Bank temporal case study complete.")
    print(f"Rows={data.x.shape[0]} features={data.x.shape[1]} hypotheses={len(TEMPORAL_HYPOTHESES)}")
    print(f"Accepted={len(accepted)} rejected={len(rejected)}")
    print(f"Wrote {RESULTS / 'bank_temporal_evidence_cards.csv'}")


if __name__ == "__main__":
    main()
