"""Reviewer-v2 audit experiments for VeraDM.

This script adds small, targeted diagnostics requested by the second review:
label-randomization on Bank Marketing, a compiler gold-set check, and paired
statistics for the human audit response file.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_temporal_case_study import TEMPORAL_HYPOTHESES, load_bank_temporal  # noqa: E402
from verifiable_discovery import Dataset, compile_hypothesis, verify_hypotheses  # noqa: E402


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


def ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean()) if len(arr) else 0.0
    if len(arr) <= 1:
        return mean, mean, mean
    half = 1.96 * float(arr.std(ddof=1)) / math.sqrt(len(arr))
    return mean, mean - half, mean + half


def run_label_randomization(seeds: int = 50) -> None:
    data = load_bank_temporal()
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(91000 + seed)
        randomized = Dataset(
            x=data.x,
            y=rng.permutation(data.y),
            split=data.split,
            feature_names=data.feature_names,
        )
        cards = verify_hypotheses(
            randomized,
            TEMPORAL_HYPOTHESES,
            permutation_rounds=200,
            bootstrap_rounds=120,
            min_support=0.025,
            min_effect=0.035,
            alpha=0.10,
        )
        accepted = sum(card.decision == "accepted" for card in cards)
        rows.append(
            {
                "seed": seed,
                "dataset": "BankMarketingMonth-label-permuted",
                "hypotheses": len(TEMPORAL_HYPOTHESES),
                "accepted": accepted,
                "accepted_rate": accepted / len(TEMPORAL_HYPOTHESES),
            }
        )
    write_csv(RESULTS / "label_randomization_audit_runs.csv", rows)
    vals = [float(row["accepted_rate"]) for row in rows]
    mean, low, high = ci95(vals)
    write_csv(
        RESULTS / "label_randomization_audit_summary.csv",
        [
            {
                "dataset": "BankMarketingMonth",
                "seeds": seeds,
                "hypotheses": len(TEMPORAL_HYPOTHESES),
                "accepted_mean": float(np.mean([row["accepted"] for row in rows])),
                "accepted_rate_mean": mean,
                "accepted_rate_ci_low": low,
                "accepted_rate_ci_high": high,
                "max_accepted": int(max(row["accepted"] for row in rows)),
            }
        ],
    )


def expected_predicates(text: str) -> set[tuple[int, str, float]]:
    lhs = text.lower().split(" then ", 1)[0].replace("if ", "", 1)
    out = set()
    for part in lhs.split(" and "):
        match = re.search(r"x(\d+)\s*(>=|<=|>|<|=)?\s*([+-]?\d+(?:\.\d+)?)?", part)
        if not match:
            continue
        feature = int(match.group(1))
        op = match.group(2) or "="
        value = float(match.group(3) or 1.0)
        out.add((feature, op, value))
    return out


def run_compiler_gold_set() -> None:
    feature_names = tuple(f"x{i}" for i in range(40))
    valid = []
    for i in range(20):
        valid.append(f"IF x{i} = 1 THEN y = 1 increases")
        valid.append(f"IF x{i} > {i % 5 + 1} THEN y = 1 decreases")
    for i in range(20):
        valid.append(f"IF x{i} = 1 AND x{(i + 7) % 40} >= 2 THEN y = 1 increases")
    invalid = [
        "x0 equals one and y increases",
        "IF x99 = 1 THEN y = 1 increases",
        "IF x1 = 1 THEN y changes",
        "IF color is red THEN y = 1 increases",
        "IF x2 around 5 THEN y = 1 decreases",
    ] * 4

    rows = []
    tp = fp = fn = 0
    direction_correct = 0
    compiled_valid = 0
    rejected_invalid = 0
    for text in valid:
        gold = expected_predicates(text)
        direction = -1 if "decreases" in text else 1
        try:
            hyp = compile_hypothesis(text, feature_names)
            pred = {(p.feature, p.op, p.value) for p in hyp.predicates}
            compiled_valid += 1
            direction_correct += int(hyp.direction == direction)
        except ValueError:
            pred = set()
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        rows.append({"text": text, "kind": "valid", "compiled": int(bool(pred)), "predicate_f1": ""})
    for text in invalid:
        try:
            compile_hypothesis(text, feature_names)
            rejected = 0
        except ValueError:
            rejected = 1
        rejected_invalid += rejected
        rows.append({"text": text, "kind": "invalid", "compiled": int(not rejected), "predicate_f1": ""})

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    write_csv(RESULTS / "compiler_gold_set_items.csv", rows)
    write_csv(
        RESULTS / "compiler_gold_set_summary.csv",
        [
            {
                "valid_items": len(valid),
                "invalid_items": len(invalid),
                "valid_compile_rate": compiled_valid / len(valid),
                "invalid_rejection_rate": rejected_invalid / len(invalid),
                "predicate_precision": precision,
                "predicate_recall": recall,
                "predicate_f1": f1,
                "direction_accuracy": direction_correct / len(valid),
            }
        ],
    )


def exact_sign_pvalue(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    lower = sum(math.comb(trials, k) for k in range(0, successes + 1)) / (2**trials)
    upper = sum(math.comb(trials, k) for k in range(successes, trials + 1)) / (2**trials)
    return min(1.0, 2.0 * min(lower, upper))


def run_human_audit_stats() -> None:
    response_path = RESULTS / "human_audit_responses.csv"
    if not response_path.exists():
        response_path = RESULTS / "human_audit_response_template.csv"
    responses = pd.read_csv(response_path)
    key = pd.read_csv(RESULTS / "human_audit_answer_key.csv")
    frame = responses.merge(key, on="item_id", how="left")
    frame = frame[frame["response_action"].notna() & (frame["response_action"].astype(str).str.len() > 0)].copy()
    frame["correct"] = frame["response_action"].eq(frame["correct_action"]).astype(float)
    paired = frame.pivot_table(index="participant_id", columns="condition", values="correct", aggfunc="mean")
    paired = paired.dropna(subset=["evidence_card", "prose_only"])
    diffs = (paired["evidence_card"] - paired["prose_only"]).astype(float).to_numpy()
    mean, low, high = ci95(diffs.tolist())
    positive = int((diffs > 0).sum())
    negative = int((diffs < 0).sum())
    nonzero = positive + negative
    write_csv(
        RESULTS / "human_audit_paired_stats.csv",
        [
            {
                "participants": len(paired),
                "mean_within_subject_difference": mean,
                "ci_low": low,
                "ci_high": high,
                "positive_differences": positive,
                "negative_differences": negative,
                "ties": int((diffs == 0).sum()),
                "sign_test_p": exact_sign_pvalue(positive, nonzero),
                "response_file": response_path.name,
            }
        ],
    )


def main() -> None:
    run_label_randomization()
    run_compiler_gold_set()
    run_human_audit_stats()
    print("Wrote review-v2 audit diagnostics")


if __name__ == "__main__":
    main()
