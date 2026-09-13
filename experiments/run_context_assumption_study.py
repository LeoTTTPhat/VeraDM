"""Two-step hypothesis + identifying-assumption proposal study.

The verifier can now accept proposer-specified context variables for the
matched-strata contrast gate. This script uses cached, domain-aware proposal
rules to emulate the second LLM step without making an API call: for each
hypothesis, the proposer returns context variables, an identifying-assumption
sentence, and a confidence score. The statistical verifier is otherwise
unchanged.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_bike_sharing_case import FEATURE_FAMILIES as BIKE_FAMILIES  # noqa: E402
from run_bike_sharing_case import HYPOTHESES as BIKE_HYPOTHESES  # noqa: E402
from run_bike_sharing_case import load_dataset as load_bike_dataset  # noqa: E402
from run_healthcare_readmission_case import FEATURE_FAMILIES as HEALTH_FAMILIES  # noqa: E402
from run_healthcare_readmission_case import HYPOTHESES as HEALTH_HYPOTHESES  # noqa: E402
from run_healthcare_readmission_case import load_dataset as load_health_dataset  # noqa: E402
from run_temporal_case_study import FEATURE_FAMILIES as BANK_FAMILIES  # noqa: E402
from run_temporal_case_study import TEMPORAL_HYPOTHESES as BANK_HYPOTHESES  # noqa: E402
from run_temporal_case_study import load_bank_temporal  # noqa: E402
from verifiable_discovery import Dataset, compile_hypothesis, render_query, verify_hypotheses  # noqa: E402


RESULTS = ROOT / "results"


DOMAIN_CONTEXT_PRIORITIES = {
    "Bank": ["demographic", "credit", "channel", "campaign", "history", "macro", "calendar"],
    "Bike": ["season", "calendar", "hour", "weather", "weather+hour"],
    "Healthcare": ["demographic", "history", "utilization", "admission", "discharge", "diagnosis", "medication", "lab"],
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


def context_proposals(
    dataset: str,
    data: Dataset,
    hypotheses: list[str],
    family_map: dict[str, str],
) -> tuple[dict[str, tuple[str, ...]], list[dict[str, object]]]:
    by_family: dict[str, list[str]] = {}
    for feature in data.feature_names:
        by_family.setdefault(family_map.get(feature, "other"), []).append(feature)

    context_map: dict[str, tuple[str, ...]] = {}
    rows: list[dict[str, object]] = []
    priorities = DOMAIN_CONTEXT_PRIORITIES[dataset]

    for text in hypotheses:
        hypothesis = compile_hypothesis(text, data.feature_names)
        query = render_query(hypothesis, data.feature_names)
        predicate_features = {data.feature_names[predicate.feature] for predicate in hypothesis.predicates}
        predicate_families = {family_map.get(feature, "other") for feature in predicate_features}

        selected: list[str] = []
        for family in priorities:
            if family in predicate_families:
                continue
            for feature in by_family.get(family, []):
                if feature not in predicate_features and feature not in selected:
                    selected.append(feature)
                    break
            if len(selected) >= 3:
                break
        context = tuple(selected[:3])
        context_map[text] = context
        confidence = 0.88 if len(context) == 3 else 0.72 if len(context) == 2 else 0.55
        rows.append(
            {
                "dataset": dataset,
                "hypothesis": text,
                "query": query,
                "predicate_families": "+".join(sorted(predicate_families)),
                "context_variables": ";".join(context),
                "context_compile_success": int(bool(context)),
                "assumption": "Within exact-match strata of "
                + ", ".join(context)
                + ", the antecedent is compared against locally similar records.",
                "llm_confidence": confidence,
            }
        )
    return context_map, rows


def summarize_cards(dataset: str, mode: str, cards) -> dict[str, object]:
    rejected = [card for card in cards if card.decision == "rejected"]
    accepted = [card for card in cards if card.decision == "accepted"]
    return {
        "dataset": dataset,
        "mode": mode,
        "hypotheses": len(cards),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "contrast_rejections": sum("counterfactual" in card.reason for card in rejected),
        "mean_context_variables": sum(len(card.context_features) for card in cards) / max(len(cards), 1),
        "accepted_with_context": sum(bool(card.context_features) for card in accepted),
    }


def run_one(
    dataset: str,
    data: Dataset,
    hypotheses: list[str],
    family_map: dict[str, str],
    *,
    min_support: float,
    min_effect: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    context_map, proposal_rows = context_proposals(dataset, data, hypotheses, family_map)
    default_cards = verify_hypotheses(
        data,
        hypotheses,
        min_support=min_support,
        alpha=0.10,
        min_effect=min_effect,
        permutation_rounds=400,
        bootstrap_rounds=300,
    )
    context_cards = verify_hypotheses(
        data,
        hypotheses,
        min_support=min_support,
        alpha=0.10,
        min_effect=min_effect,
        permutation_rounds=400,
        bootstrap_rounds=300,
        context_feature_map=context_map,
    )
    rows = [
        summarize_cards(dataset, "fixed-default-context", default_cards),
        summarize_cards(dataset, "two-step-context-proposer", context_cards),
    ]
    rows[1]["context_compile_rate"] = sum(r["context_compile_success"] for r in proposal_rows) / len(proposal_rows)
    rows[1]["unique_context_sets"] = len({r["context_variables"] for r in proposal_rows})
    rows[0]["context_compile_rate"] = ""
    rows[0]["unique_context_sets"] = ""
    return proposal_rows, rows


def main() -> None:
    bank = load_bank_temporal()
    bike, _ = load_bike_dataset()
    health = load_health_dataset()

    proposal_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for proposals, summary in [
        run_one(
            "Bank",
            bank,
            BANK_HYPOTHESES,
            BANK_FAMILIES,
            min_support=0.025,
            min_effect=0.035,
        ),
        run_one(
            "Bike",
            bike,
            BIKE_HYPOTHESES,
            BIKE_FAMILIES,
            min_support=0.025,
            min_effect=0.050,
        ),
        run_one(
            "Healthcare",
            health,
            HEALTH_HYPOTHESES,
            HEALTH_FAMILIES,
            min_support=0.025,
            min_effect=0.025,
        ),
    ]:
        proposal_rows.extend(proposals)
        summary_rows.extend(summary)

    write_csv(RESULTS / "context_assumption_proposals.csv", proposal_rows)
    write_csv(RESULTS / "context_assumption_summary.csv", summary_rows)
    print(f"Wrote {RESULTS / 'context_assumption_proposals.csv'}")
    print(f"Wrote {RESULTS / 'context_assumption_summary.csv'}")


if __name__ == "__main__":
    main()
