"""Compiler benchmark for temporal, sequence, and treatment-effect predicates."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verifiable_discovery import Dataset, compile_hypothesis, render_query, sigmoid, verify_hypotheses  # noqa: E402


RESULTS = ROOT / "results"


FEATURE_NAMES = (
    "x0",  # treatment
    "x1",  # subgroup
    "x2",  # source event
    "x3",  # follow-up event
    "event_x2_within_30d",
    "seq_x2_x3_within_7d",
    "x6",
    "x7",
)


VALID = [
    ("tabular", "IF x6 = 1 AND x7 >= 1 THEN y = 1 decreases"),
    ("temporal", "IF EVENT x2 WITHIN 30 DAYS THEN y = 1 increases"),
    ("sequence", "IF SEQUENCE x2 -> x3 WITHIN 7 DAYS THEN y = 1 increases"),
    ("treatment", "IF TREATMENT x0 = 1 AMONG x1 = 1 THEN y = 1 increases"),
    ("treatment", "IF TREAT x0 = 1 FOR x1 = 1 AND x6 = 1 THEN y = 1 increases"),
]

INVALID = [
    ("temporal", "IF EVENT x2 WITHIN 90 DAYS THEN y = 1 increases"),
    ("sequence", "IF SEQUENCE x2 -> x8 WITHIN 7 DAYS THEN y = 1 increases"),
    ("treatment", "IF TREATMENT x0 = 1 THEN y = 1 increases"),
    ("treatment", "IF TREATMENT x99 = 1 AMONG x1 = 1 THEN y = 1 increases"),
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_language_data(n_per_split: int = 3000, seed: int = 97) -> Dataset:
    rng = np.random.default_rng(seed)
    xs, ys, splits = [], [], []
    for split in (0, 1, 2):
        x = rng.binomial(1, 0.5, size=(n_per_split, len(FEATURE_NAMES)))
        x[:, 4] = x[:, 2] & rng.binomial(1, 0.70, n_per_split)
        x[:, 5] = x[:, 2] & x[:, 3] & rng.binomial(1, 0.75, n_per_split)
        subgroup = x[:, 1].astype(bool)
        treated = x[:, 0].astype(bool)
        logit = (
            -1.35
            + 0.95 * x[:, 4]
            + 1.20 * x[:, 5]
            + 1.10 * (subgroup & treated)
            - 0.75 * (x[:, 6] & x[:, 7])
            + 0.30 * subgroup
        )
        y = rng.binomial(1, sigmoid(logit))
        xs.append(x)
        ys.append(y)
        splits.append(np.full(n_per_split, split))
    return Dataset(np.vstack(xs), np.concatenate(ys), np.concatenate(splits), FEATURE_NAMES)


def main() -> None:
    rows: list[dict[str, object]] = []
    compiled_valid = 0
    rejected_invalid = 0
    for kind, text in VALID:
        try:
            hyp = compile_hypothesis(text, FEATURE_NAMES)
            compiled = 1
            compiled_valid += 1
            query = render_query(hyp, FEATURE_NAMES)
            estimand = hyp.estimand
        except ValueError as exc:
            compiled = 0
            query = str(exc)
            estimand = ""
        rows.append({"kind": kind, "text": text, "compiled": compiled, "query": query, "estimand": estimand})

    for kind, text in INVALID:
        try:
            hyp = compile_hypothesis(text, FEATURE_NAMES)
            rejected = 0
            query = render_query(hyp, FEATURE_NAMES)
        except ValueError as exc:
            rejected = 1
            query = str(exc)
        rejected_invalid += rejected
        rows.append({"kind": kind, "text": text, "compiled": int(not rejected), "query": query, "estimand": ""})

    data = make_language_data()
    cards = verify_hypotheses(
        data,
        [text for _, text in VALID],
        min_support=0.04,
        alpha=0.10,
        min_effect=0.03,
        permutation_rounds=400,
        bootstrap_rounds=300,
    )
    card_by_text = {card.hypothesis: card for card in cards}
    for row in rows:
        card = card_by_text.get(str(row["text"]))
        if card:
            row.update(
                {
                    "decision": card.decision,
                    "validation_effect": card.validation_effect,
                    "holdout_effect": card.holdout_effect,
                    "q_value": card.q_value,
                    "reason": card.reason,
                }
            )

    summary = [
        {
            "valid_items": len(VALID),
            "invalid_items": len(INVALID),
            "valid_compile_rate": compiled_valid / len(VALID),
            "invalid_rejection_rate": rejected_invalid / len(INVALID),
            "verified_items": len(cards),
            "accepted_verified": sum(card.decision == "accepted" for card in cards),
            "language_kinds": len({kind for kind, _ in VALID}),
            "treatment_items": sum(kind == "treatment" for kind, _ in VALID),
        }
    ]
    write_csv(RESULTS / "language_expansion_items.csv", rows)
    write_csv(RESULTS / "language_expansion_summary.csv", summary)
    print(f"Wrote {RESULTS / 'language_expansion_items.csv'}")
    print(f"Wrote {RESULTS / 'language_expansion_summary.csv'}")


if __name__ == "__main__":
    main()
