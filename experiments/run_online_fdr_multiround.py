"""Multi-round verdict-adaptive discovery with alpha-wealth spending.

This experiment mirrors the online-FDR wealth-spending result. A proposer may
condition later rounds on earlier verifier verdicts, but each round receives a
fresh confirmatory held-out shard and spends a pre-visible part of a finite
alpha budget. Acceptance uses conservative Bonferroni tests on the round shard.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_bike_sharing_case import FEATURE_FAMILIES as BIKE_FAMILIES  # noqa: E402
from run_bike_sharing_case import HYPOTHESES as BIKE_HYPOTHESES  # noqa: E402
from run_bike_sharing_case import load_dataset as load_bike_dataset  # noqa: E402
from run_temporal_case_study import FEATURE_FAMILIES as BANK_FAMILIES  # noqa: E402
from run_temporal_case_study import TEMPORAL_HYPOTHESES as BANK_HYPOTHESES  # noqa: E402
from run_temporal_case_study import load_bank_temporal  # noqa: E402
from verifiable_discovery import (  # noqa: E402
    Dataset,
    compile_hypothesis,
    matched_strata_effect,
    pattern_indicator,
    permutation_p_value,
    render_query,
    risk_difference,
    support,
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


def hypothesis_families(text: str, data: Dataset, family_map: dict[str, str]) -> tuple[str, ...]:
    hypothesis = compile_hypothesis(text, data.feature_names)
    families = []
    for predicate in hypothesis.predicates:
        feature = data.feature_names[predicate.feature]
        families.append(family_map.get(feature, "other"))
    return tuple(sorted(set(families)))


def choose_batch(
    remaining: list[str],
    data: Dataset,
    family_map: dict[str, str],
    accepted_families: set[str],
    rejected_families: set[str],
    *,
    round_id: int,
    max_batch: int = 8,
) -> tuple[list[str], str]:
    if round_id == 1:
        pool = [text for text in remaining if " AND " not in text]
        rationale = "schema-broad singleton scan"
    elif accepted_families:
        pool = [
            text
            for text in remaining
            if accepted_families.intersection(hypothesis_families(text, data, family_map))
        ]
        rationale = "follow-up around previously accepted families"
    elif rejected_families:
        pool = [
            text
            for text in remaining
            if rejected_families.intersection(hypothesis_families(text, data, family_map))
        ]
        rationale = "repair around previously rejected families"
    else:
        pool = remaining
        rationale = "fallback unexplored scan"

    if len(pool) < max_batch:
        seen = set(pool)
        pool = pool + [text for text in remaining if text not in seen]
    return pool[:max_batch], rationale


def evaluate_round(
    data: Dataset,
    batch: list[str],
    holdout_indices: np.ndarray,
    *,
    alpha_t: float,
    min_support: float,
    min_effect: float,
    seed: int,
) -> list[dict[str, object]]:
    validation_x = data.x[data.split == 1]
    validation_y = data.y[data.split == 1]
    heldout_x = data.x[holdout_indices]
    heldout_y = data.y[holdout_indices]
    threshold = alpha_t / max(len(batch), 1)
    rows: list[dict[str, object]] = []

    for offset, text in enumerate(batch):
        hypothesis = compile_hypothesis(text, data.feature_names)
        validation_active = pattern_indicator(validation_x, hypothesis.predicates)
        heldout_active = pattern_indicator(heldout_x, hypothesis.predicates)
        validation_support = support(validation_active)
        heldout_support = support(heldout_active)
        validation_effect = risk_difference(validation_y, validation_active)
        heldout_effect = risk_difference(heldout_y, heldout_active)
        contrast_effect = matched_strata_effect(validation_x, validation_y, hypothesis)
        p_holdout = permutation_p_value(
            heldout_y,
            heldout_active,
            hypothesis.direction,
            rounds=500,
            seed=seed + offset,
        )
        signed_validation = hypothesis.direction * validation_effect
        signed_heldout = hypothesis.direction * heldout_effect
        signed_contrast = hypothesis.direction * contrast_effect

        reasons = []
        if validation_support < min_support or heldout_support < min_support:
            reasons.append("support")
        if signed_validation < min_effect:
            reasons.append("validation")
        if signed_contrast < min_effect:
            reasons.append("contrast")
        if signed_heldout < min_effect:
            reasons.append("heldout")
        if p_holdout > threshold:
            reasons.append("bonferroni")

        rows.append(
            {
                "hypothesis": text,
                "query": render_query(hypothesis, data.feature_names),
                "decision": "accepted" if not reasons else "rejected",
                "validation_support": validation_support,
                "heldout_support": heldout_support,
                "validation_effect": validation_effect,
                "contrast_effect": contrast_effect,
                "heldout_effect": heldout_effect,
                "p_holdout": p_holdout,
                "alpha_per_test": threshold,
                "reason": ";".join(reasons) if reasons else "all gates passed",
            }
        )
    return rows


def run_dataset(
    *,
    name: str,
    data: Dataset,
    hypotheses: list[str],
    family_map: dict[str, str],
    min_support: float,
    min_effect: float,
    alpha_total: float = 0.10,
    schedule: tuple[float, ...] = (0.50, 0.30, 0.20),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    heldout_indices = np.where(data.split == 2)[0]
    shards = np.array_split(heldout_indices, len(schedule))
    remaining = list(hypotheses)
    accepted_families: set[str] = set()
    rejected_families: set[str] = set()
    wealth = alpha_total
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for round_id, gamma in enumerate(schedule, start=1):
        batch, rationale = choose_batch(
            remaining,
            data,
            family_map,
            accepted_families,
            rejected_families,
            round_id=round_id,
        )
        if not batch:
            continue
        alpha_t = alpha_total * gamma
        wealth_before = wealth
        wealth = max(0.0, wealth - alpha_t)
        round_rows = evaluate_round(
            data,
            batch,
            shards[round_id - 1],
            alpha_t=alpha_t,
            min_support=min_support,
            min_effect=min_effect,
            seed=3100 + 97 * round_id + len(name),
        )
        remaining = [text for text in remaining if text not in set(batch)]
        for row in round_rows:
            families = hypothesis_families(str(row["hypothesis"]), data, family_map)
            if row["decision"] == "accepted":
                accepted_families.update(families)
            else:
                rejected_families.update(families)
            detail_rows.append(
                {
                    "dataset": name,
                    "round": round_id,
                    "alpha_spent": alpha_t,
                    "wealth_before": wealth_before,
                    "wealth_after": wealth,
                    "heldout_shard_n": int(len(shards[round_id - 1])),
                    "proposer_conditioning": rationale,
                    "families": "+".join(families),
                    **row,
                }
            )

        accepted = [row for row in round_rows if row["decision"] == "accepted"]
        summary_rows.append(
            {
                "dataset": name,
                "round": round_id,
                "alpha_spent": alpha_t,
                "wealth_after": wealth,
                "heldout_shard_n": int(len(shards[round_id - 1])),
                "proposed": len(round_rows),
                "accepted": len(accepted),
                "rejected": len(round_rows) - len(accepted),
                "conditioned_on": rationale,
                "accepted_queries": " | ".join(str(row["query"]) for row in accepted[:3]),
            }
        )

    return detail_rows, summary_rows


def main() -> None:
    bank_data = load_bank_temporal()
    bike_data, _ = load_bike_dataset()
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for rows, summary in [
        run_dataset(
            name="Bank",
            data=bank_data,
            hypotheses=BANK_HYPOTHESES,
            family_map=BANK_FAMILIES,
            min_support=0.025,
            min_effect=0.035,
        ),
        run_dataset(
            name="Bike",
            data=bike_data,
            hypotheses=BIKE_HYPOTHESES,
            family_map=BIKE_FAMILIES,
            min_support=0.025,
            min_effect=0.050,
        ),
    ]:
        detail_rows.extend(rows)
        summary_rows.extend(summary)

    write_csv(RESULTS / "online_fdr_multiround_runs.csv", detail_rows)
    write_csv(RESULTS / "online_fdr_multiround_summary.csv", summary_rows)
    print(f"Wrote {RESULTS / 'online_fdr_multiround_runs.csv'}")
    print(f"Wrote {RESULTS / 'online_fdr_multiround_summary.csv'}")


if __name__ == "__main__":
    main()
