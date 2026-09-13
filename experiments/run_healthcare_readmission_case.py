"""Healthcare readmission held-out replication case study.

Dataset: UCI Diabetes 130-US hospitals for years 1999--2008. The dataset does
not expose exact dates, but encounter_id is monotonically assigned in the source
and is commonly treated as an encounter-order proxy. We split by encounter_id
tertiles: early discovery, middle validation, late held-out replication.
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
URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"


HYPOTHESES = [
    "IF x0 = 1 THEN y = 1 increases",
    "IF x1 = 1 THEN y = 1 increases",
    "IF x2 = 1 THEN y = 1 increases",
    "IF x3 = 1 THEN y = 1 increases",
    "IF x4 = 1 THEN y = 1 increases",
    "IF x5 = 1 THEN y = 1 increases",
    "IF x6 = 1 THEN y = 1 increases",
    "IF x7 = 1 THEN y = 1 increases",
    "IF x8 = 1 THEN y = 1 increases",
    "IF x9 = 1 THEN y = 1 increases",
    "IF x10 = 1 THEN y = 1 increases",
    "IF x11 = 1 THEN y = 1 increases",
    "IF x12 = 1 THEN y = 1 increases",
    "IF x13 = 1 THEN y = 1 increases",
    "IF x14 = 1 THEN y = 1 decreases",
    "IF x15 = 1 THEN y = 1 decreases",
    "IF x16 = 1 THEN y = 1 increases",
    "IF x17 = 1 THEN y = 1 increases",
    "IF x18 = 1 THEN y = 1 increases",
    "IF x19 = 1 THEN y = 1 decreases",
    "IF x20 = 1 THEN y = 1 increases",
    "IF x21 = 1 THEN y = 1 increases",
    "IF x22 = 1 THEN y = 1 increases",
    "IF x23 = 1 THEN y = 1 increases",
    "IF x24 = 1 THEN y = 1 increases",
    "IF x25 = 1 THEN y = 1 increases",
    "IF x26 = 1 THEN y = 1 decreases",
    "IF x27 = 1 THEN y = 1 increases",
    "IF x28 = 1 THEN y = 1 increases",
    "IF x29 = 1 THEN y = 1 increases",
    "IF x30 = 1 THEN y = 1 increases",
    "IF x31 = 1 THEN y = 1 increases",
    "IF x7 = 1 AND x13 = 1 THEN y = 1 increases",
    "IF x7 = 1 AND x21 = 1 THEN y = 1 increases",
    "IF x6 = 1 AND x13 = 1 THEN y = 1 increases",
    "IF x5 = 1 AND x7 = 1 THEN y = 1 increases",
    "IF x1 = 1 AND x4 = 1 THEN y = 1 increases",
    "IF x14 = 1 AND x15 = 1 THEN y = 1 decreases",
]


FEATURE_FAMILIES = {
    "diab_age_ge_70": "demographic",
    "diab_time_in_hospital_ge_7": "utilization",
    "diab_num_lab_procedures_ge_60": "utilization",
    "diab_num_procedures_zero": "utilization",
    "diab_num_medications_ge_20": "medication",
    "diab_outpatient_prior": "history",
    "diab_emergency_prior": "history",
    "diab_inpatient_prior": "history",
    "diab_diagnoses_ge_8": "diagnosis",
    "diab_insulin_changed": "medication",
    "diab_diabetes_med_yes": "medication",
    "diab_change_yes": "medication",
    "diab_discharge_home": "discharge",
    "diab_admission_emergency": "admission",
    "diab_no_prior_inpatient": "history",
    "diab_short_stay": "utilization",
    "diab_age_ge_80": "demographic",
    "diab_num_medications_ge_25": "medication",
    "diab_num_procedures_ge_3": "utilization",
    "diab_no_prior_utilization": "history",
    "diab_discharge_facility": "discharge",
    "diab_admission_elective": "admission",
    "diab_a1c_missing": "lab",
    "diab_a1c_high": "lab",
    "diab_metformin_yes": "medication",
    "diab_insulin_yes": "medication",
    "diab_insulin_no": "medication",
    "diab_female": "demographic",
    "diab_race_african_american": "demographic",
    "diab_primary_circulatory": "diagnosis",
    "diab_primary_diabetes": "diagnosis",
    "diab_specialty_emergency": "specialty",
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


def fetch_diabetes() -> pd.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    cached = DATA / "diabetes_readmission.csv"
    if cached.exists():
        return pd.read_csv(cached)
    with urlopen(URL, timeout=120) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_name = [name for name in zf.namelist() if name.endswith("diabetic_data.csv")][0]
        with zf.open(csv_name) as f:
            frame = pd.read_csv(f)
    frame.to_csv(cached, index=False)
    return frame


def load_dataset() -> Dataset:
    frame = fetch_diabetes().replace("?", np.nan).copy()
    frame = frame.sort_values("encounter_id").reset_index(drop=True)
    target = frame["readmitted"].eq("<30").astype(int)
    age_num = frame["age"].str.extract(r"\[(\d+)-").astype(float)[0].fillna(0)
    procedures = pd.to_numeric(frame["num_procedures"], errors="coerce").fillna(0)
    outpatient = pd.to_numeric(frame["number_outpatient"], errors="coerce").fillna(0)
    emergency = pd.to_numeric(frame["number_emergency"], errors="coerce").fillna(0)
    inpatient = pd.to_numeric(frame["number_inpatient"], errors="coerce").fillna(0)
    diag1_num = pd.to_numeric(frame["diag_1"], errors="coerce")
    features = pd.DataFrame(
        {
            "diab_age_ge_70": age_num >= 70,
            "diab_time_in_hospital_ge_7": pd.to_numeric(frame["time_in_hospital"], errors="coerce") >= 7,
            "diab_num_lab_procedures_ge_60": pd.to_numeric(frame["num_lab_procedures"], errors="coerce") >= 60,
            "diab_num_procedures_zero": procedures.eq(0),
            "diab_num_medications_ge_20": pd.to_numeric(frame["num_medications"], errors="coerce") >= 20,
            "diab_outpatient_prior": outpatient > 0,
            "diab_emergency_prior": emergency > 0,
            "diab_inpatient_prior": inpatient > 0,
            "diab_diagnoses_ge_8": pd.to_numeric(frame["number_diagnoses"], errors="coerce") >= 8,
            "diab_insulin_changed": frame["insulin"].isin(["Up", "Down"]),
            "diab_diabetes_med_yes": frame["diabetesMed"].eq("Yes"),
            "diab_change_yes": frame["change"].eq("Ch"),
            "diab_discharge_home": frame["discharge_disposition_id"].astype(str).eq("1"),
            "diab_admission_emergency": frame["admission_type_id"].astype(str).isin(["1", "2", "7"]),
            "diab_no_prior_inpatient": inpatient.eq(0),
            "diab_short_stay": pd.to_numeric(frame["time_in_hospital"], errors="coerce") <= 2,
            "diab_age_ge_80": age_num >= 80,
            "diab_num_medications_ge_25": pd.to_numeric(frame["num_medications"], errors="coerce") >= 25,
            "diab_num_procedures_ge_3": procedures >= 3,
            "diab_no_prior_utilization": (outpatient + emergency + inpatient).eq(0),
            "diab_discharge_facility": frame["discharge_disposition_id"].astype(str).isin(["2", "3", "4", "5", "22", "23", "24"]),
            "diab_admission_elective": frame["admission_type_id"].astype(str).eq("3"),
            "diab_a1c_missing": frame["A1Cresult"].isna(),
            "diab_a1c_high": frame["A1Cresult"].isin([">7", ">8"]),
            "diab_metformin_yes": ~frame["metformin"].eq("No"),
            "diab_insulin_yes": ~frame["insulin"].eq("No"),
            "diab_insulin_no": frame["insulin"].eq("No"),
            "diab_female": frame["gender"].eq("Female"),
            "diab_race_african_american": frame["race"].eq("AfricanAmerican"),
            "diab_primary_circulatory": diag1_num.between(390, 459) | frame["diag_1"].astype(str).str.startswith("V"),
            "diab_primary_diabetes": frame["diag_1"].astype(str).str.startswith("250"),
            "diab_specialty_emergency": frame["medical_specialty"].eq("Emergency/Trauma"),
        }
    )
    n = len(frame)
    split = np.zeros(n, dtype=int)
    split[int(n / 3) : int(2 * n / 3)] = 1
    split[int(2 * n / 3) :] = 2
    return Dataset(features.astype(int).to_numpy(), target.to_numpy(), split, tuple(features.columns))


def case_summary(data: Dataset) -> list[dict[str, object]]:
    rows = []
    names = {0: "early_discovery", 1: "middle_validation", 2: "late_holdout"}
    for split_id, name in names.items():
        mask = data.split == split_id
        rows.append(
            {
                "split": name,
                "n": int(mask.sum()),
                "readmit_rate": float(data.y[mask].mean()),
                "hypotheses": len(HYPOTHESES),
                "features": len(data.feature_names),
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
                "dataset": "DiabetesReadmission",
                "split": "early/middle/late encounter_id",
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
    data = load_dataset()
    cards = verify_hypotheses(
        data,
        HYPOTHESES,
        min_support=0.04,
        min_effect=0.01,
        alpha=0.10,
        permutation_rounds=500,
        bootstrap_rounds=500,
    )
    rows = evidence_rows(cards)
    write_csv(RESULTS / "healthcare_readmission_evidence_cards.csv", rows)
    write_csv(RESULTS / "healthcare_readmission_case_summary.csv", case_summary(data))
    accepted = [r for r in rows if r["decision"] == "accepted"]
    print("Healthcare readmission case study complete.")
    print(f"Rows={data.x.shape[0]} features={data.x.shape[1]} hypotheses={len(HYPOTHESES)} accepted={len(accepted)}")
    print(f"Wrote {RESULTS / 'healthcare_readmission_evidence_cards.csv'}")


if __name__ == "__main__":
    main()
