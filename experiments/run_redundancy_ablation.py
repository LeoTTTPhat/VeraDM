"""Measure accepted-card redundancy before and after pruning."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verifiable_discovery import (  # noqa: E402
    Dataset,
    compile_hypothesis,
    discovery_split,
    pattern_indicator,
    prune_redundant_cards,
    verify_hypotheses,
)
sys.path.insert(0, str(ROOT / "experiments"))
from run_verifiable_discovery import BASELINE_CONFIGS, load_adult_income, load_german_credit  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def avg_pairwise_jaccard(data: Dataset, hypotheses: list[str]) -> float:
    if len(hypotheses) < 2:
        return 0.0
    x, _ = discovery_split(data, 1)
    active_sets = []
    for text in hypotheses:
        try:
            h = compile_hypothesis(text, data.feature_names)
        except ValueError:
            continue
        active_sets.append(pattern_indicator(x, h.predicates))
    if len(active_sets) < 2:
        return 0.0
    values = []
    for i in range(len(active_sets)):
        for j in range(i + 1, len(active_sets)):
            union = np.logical_or(active_sets[i], active_sets[j]).sum()
            values.append(0.0 if union == 0 else float(np.logical_and(active_sets[i], active_sets[j]).sum() / union))
    return float(np.mean(values)) if values else 0.0


def run_dataset(name: str, data: Dataset, proposals: list[str]) -> dict[str, object]:
    cards = verify_hypotheses(
        data,
        proposals,
        permutation_rounds=220,
        bootstrap_rounds=220,
        **BASELINE_CONFIGS["FullVerifier"],
    )
    accepted_before = [card for card in cards if card.decision == "accepted"]
    pruned = prune_redundant_cards(data, cards)
    accepted_after = [card for card in pruned if card.decision == "accepted"]
    before_overlap = avg_pairwise_jaccard(data, [card.hypothesis for card in accepted_before])
    after_overlap = avg_pairwise_jaccard(data, [card.hypothesis for card in accepted_after])
    return {
        "dataset": name,
        "accepted_before": len(accepted_before),
        "accepted_after": len(accepted_after),
        "removed": len(accepted_before) - len(accepted_after),
        "avg_jaccard_before": before_overlap,
        "avg_jaccard_after": after_overlap,
        "overlap_reduction": before_overlap - after_overlap,
    }


def main() -> None:
    rows = []
    for name, loader in [("AdultIncome", load_adult_income), ("GermanCredit", load_german_credit)]:
        data = loader()
        proposal_path = sorted((ROOT / "results" / "llm_proposals").glob(f"{name}_openai_*.json"))
        if not proposal_path:
            continue
        import json

        proposals = json.loads(proposal_path[-1].read_text())["hypotheses"]
        rows.append(run_dataset(name, data, proposals))
    out = ROOT / "results" / "redundancy_ablation.csv"
    write_csv(out, rows)
    print(f"Wrote {out}")
    for row in rows:
        print(
            f"{row['dataset']}: {row['accepted_before']} -> {row['accepted_after']} "
            f"overlap {row['avg_jaccard_before']:.3f}->{row['avg_jaccard_after']:.3f}"
        )


if __name__ == "__main__":
    main()
