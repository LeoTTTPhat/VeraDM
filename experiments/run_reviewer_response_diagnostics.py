"""Targeted implementation diagnostics for the VeraDM verifier.

The goal is not to introduce new methodology, but to make the implementation
choices inspectable: threshold protocol, dependence-robust correction, runtime /
permutation sensitivity, redundancy-aware baseline scoring, and balance behavior
of the matched-contrast gate.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_bike_sharing_case import HYPOTHESES as BIKE_HYPOTHESES  # noqa: E402
from run_bike_sharing_case import load_dataset as load_bike_dataset  # noqa: E402
from run_healthcare_readmission_case import HYPOTHESES as HEALTH_HYPOTHESES  # noqa: E402
from run_healthcare_readmission_case import load_dataset as load_health_dataset  # noqa: E402
from run_temporal_case_study import TEMPORAL_HYPOTHESES as BANK_HYPOTHESES  # noqa: E402
from run_temporal_case_study import load_bank_temporal  # noqa: E402
from verifiable_discovery import (  # noqa: E402
    Dataset,
    compile_hypothesis,
    pattern_indicator,
    verify_hypotheses,
)


RESULTS = ROOT / "results"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def by_adjust(p_values: list[float]) -> list[float]:
    values = np.array(p_values, dtype=float)
    m = len(values)
    c_m = sum(1.0 / i for i in range(1, m + 1))
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = m - rank_from_end + 1
        running = min(running, values[idx] * m * c_m / rank)
        adjusted[idx] = min(running, 1.0)
    return [float(x) for x in adjusted]


def dependence_robust_counts() -> list[dict[str, object]]:
    rows = []
    for dataset, path in [
        ("Bank", RESULTS / "bank_temporal_evidence_cards.csv"),
        ("Bike", RESULTS / "bike_sharing_evidence_cards.csv"),
        ("Healthcare", RESULTS / "healthcare_readmission_evidence_cards.csv"),
    ]:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        by_q = by_adjust(frame["p_value"].astype(float).tolist())
        frame = frame.assign(by_q=by_q)
        bh_accepted = frame[frame["decision"].eq("accepted")]
        by_survivors = bh_accepted[bh_accepted["by_q"] <= 0.10]
        rows.append(
            {
                "dataset": dataset,
                "hypotheses": len(frame),
                "bh_accepted": len(bh_accepted),
                "by_survivors": len(by_survivors),
                "by_survival_rate": len(by_survivors) / max(len(bh_accepted), 1),
                "min_by_q": float(frame["by_q"].min()),
            }
        )
    return rows


def threshold_sensitivity() -> list[dict[str, object]]:
    rows = []
    settings = [
        ("support_low", 0.010, 0.035),
        ("default", 0.025, 0.035),
        ("support_high", 0.050, 0.035),
        ("effect_low", 0.025, 0.025),
        ("effect_high", 0.025, 0.050),
    ]
    for dataset, data, hypotheses, base_effect in [
        ("Bank", load_bank_temporal(), BANK_HYPOTHESES, 0.035),
        ("Bike", load_bike_dataset()[0], BIKE_HYPOTHESES, 0.050),
    ]:
        for name, support, effect in settings:
            actual_effect = effect if "effect" in name else base_effect
            cards = verify_hypotheses(
                data,
                hypotheses,
                min_support=support,
                min_effect=actual_effect,
                alpha=0.10,
                permutation_rounds=250,
                bootstrap_rounds=120,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "setting": name,
                    "min_support": support,
                    "min_effect": actual_effect,
                    "accepted": sum(card.decision == "accepted" for card in cards),
                    "support_rejections": sum("support" in card.reason for card in cards),
                    "effect_rejections": sum("effect" in card.reason for card in cards),
                }
            )
    return rows


def permutation_runtime_sensitivity() -> list[dict[str, object]]:
    rows = []
    for dataset, data, hypotheses, effect in [
        ("Bank", load_bank_temporal(), BANK_HYPOTHESES, 0.035),
        ("Bike", load_bike_dataset()[0], BIKE_HYPOTHESES, 0.050),
    ]:
        for rounds in [200, 500, 1000]:
            started = time.perf_counter()
            cards = verify_hypotheses(
                data,
                hypotheses,
                min_support=0.025,
                min_effect=effect,
                alpha=0.10,
                permutation_rounds=rounds,
                bootstrap_rounds=80,
            )
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "dataset": dataset,
                    "permutation_rounds": rounds,
                    "p_floor": 1.0 / (rounds + 1),
                    "accepted": sum(card.decision == "accepted" for card in cards),
                    "runtime_seconds": elapsed,
                    "hypotheses_per_second": len(cards) / elapsed if elapsed else 0.0,
                }
            )
    return rows


def smd(values: np.ndarray, active: np.ndarray) -> float:
    if active.sum() == 0 or (~active).sum() == 0:
        return 0.0
    p1 = values[active].mean()
    p0 = values[~active].mean()
    pooled = max((p1 * (1 - p1) + p0 * (1 - p0)) / 2.0, 1e-9)
    return float(abs(p1 - p0) / np.sqrt(pooled))


def balance_for_hypothesis(data: Dataset, text: str, split_id: int = 1) -> tuple[float, float]:
    hyp = compile_hypothesis(text, data.feature_names)
    x = data.x[data.split == split_id]
    active = pattern_indicator(x, hyp.predicates)
    predicate_features = {p.feature for p in hyp.predicates}
    context = [idx for idx in range(min(5, x.shape[1])) if idx not in predicate_features][:3]
    if not context:
        return 0.0, 1.0
    before = max(smd(x[:, idx], active) for idx in context)
    retained = 0
    for values in {tuple(row) for row in x[:, context]}:
        mask = np.all(x[:, context] == np.array(values), axis=1)
        if (active & mask).any() and ((~active) & mask).any():
            retained += int(mask.sum())
    return before, retained / len(active)


def contrast_balance_diagnostics() -> list[dict[str, object]]:
    rows = []
    for dataset, data, path in [
        ("Bank", load_bank_temporal(), RESULTS / "bank_temporal_evidence_cards.csv"),
        ("Bike", load_bike_dataset()[0], RESULTS / "bike_sharing_evidence_cards.csv"),
        ("Healthcare", load_health_dataset(), RESULTS / "healthcare_readmission_evidence_cards.csv"),
    ]:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        accepted = frame[frame["decision"].eq("accepted")]
        before_values = []
        retained_values = []
        for text in accepted["hypothesis"]:
            before, retained = balance_for_hypothesis(data, str(text))
            before_values.append(before)
            retained_values.append(retained)
        rows.append(
            {
                "dataset": dataset,
                "accepted": len(accepted),
                "mean_max_smd_before": float(np.mean(before_values)) if before_values else 0.0,
                "max_smd_after_exact_context": 0.0,
                "mean_retained_in_overlap_strata": float(np.mean(retained_values)) if retained_values else 0.0,
            }
        )
    return rows


def main() -> None:
    outputs = {
        "dependence_robust_counts.csv": dependence_robust_counts(),
        "threshold_sensitivity.csv": threshold_sensitivity(),
        "permutation_runtime_sensitivity.csv": permutation_runtime_sensitivity(),
        "contrast_balance_diagnostics.csv": contrast_balance_diagnostics(),
    }
    for filename, rows in outputs.items():
        write_csv(RESULTS / filename, rows)
        print(f"Wrote {RESULTS / filename}")


if __name__ == "__main__":
    main()
