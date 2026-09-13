"""Stress benchmark for LLM-guided verifiable discovery.

This experiment creates many true, null, rare, redundant, confounded, and shifted
hypotheses, then compares the verifier against exhaustive rule enumeration.
"""

from __future__ import annotations

import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verifiable_discovery import Dataset, verify_hypotheses  # noqa: E402


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_stress_data(n_per_split: int = 1600, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    xs, ys, splits = [], [], []
    for split in (0, 1, 2):
        x = rng.binomial(1, 0.5, size=(n_per_split, 14))
        z = rng.binomial(1, 0.5, n_per_split)
        x[:, 8] = z

        logit = -1.35 + 1.1 * x[:, 0] + 0.95 * x[:, 1] + 0.9 * (x[:, 0] & x[:, 1])
        y = rng.binomial(1, sigmoid(logit))

        # Shifted: attractive before holdout, reversed in holdout.
        if split in (0, 1):
            x[:, 2] = np.where(y == 1, rng.binomial(1, 0.9, n_per_split), rng.binomial(1, 0.12, n_per_split))
            x[:, 3] = np.where(y == 1, rng.binomial(1, 0.78, n_per_split), rng.binomial(1, 0.28, n_per_split))
        else:
            x[:, 2] = np.where(y == 1, rng.binomial(1, 0.12, n_per_split), rng.binomial(1, 0.9, n_per_split))
            x[:, 3] = np.where(y == 1, rng.binomial(1, 0.25, n_per_split), rng.binomial(1, 0.72, n_per_split))

        # Rare but strong.
        rare = rng.binomial(1, 0.035, n_per_split)
        y = np.where(rare == 1, rng.binomial(1, 0.92, n_per_split), y)
        x[:, 4] = rare

        # Redundant aliases of true signals.
        x[:, 5] = np.where(x[:, 0] == 1, rng.binomial(1, 0.92, n_per_split), rng.binomial(1, 0.08, n_per_split))
        x[:, 6] = np.where(x[:, 1] == 1, rng.binomial(1, 0.9, n_per_split), rng.binomial(1, 0.1, n_per_split))

        # Confounded proxy: associated with y through z and x0 in validation, less stable in holdout.
        if split in (0, 1):
            x[:, 7] = np.where((z | x[:, 0]).astype(bool), rng.binomial(1, 0.72, n_per_split), rng.binomial(1, 0.25, n_per_split))
        else:
            x[:, 7] = rng.binomial(1, 0.5, n_per_split)

        # Context and null features.
        x[:, 9:] = rng.binomial(1, 0.5, size=(n_per_split, 5))

        xs.append(x)
        ys.append(y)
        splits.append(np.full(n_per_split, split))

    return Dataset(np.vstack(xs), np.concatenate(ys), np.concatenate(splits), tuple(f"x{i}" for i in range(14)))


def hypothesis(features: tuple[int, ...], direction: str = "increases") -> str:
    antecedent = " AND ".join(f"x{feature} = 1" for feature in features)
    return f"IF {antecedent} THEN y = 1 {direction}"


def stress_proposals(rng: np.random.Generator, n_random_null: int = 60) -> list[str]:
    proposals = [
        hypothesis((0,)),
        hypothesis((1,)),
        hypothesis((0, 1)),
        hypothesis((2,)),
        hypothesis((3,)),
        hypothesis((4,)),
        hypothesis((5,)),
        hypothesis((6,)),
        hypothesis((5, 6)),
        hypothesis((7,)),
        hypothesis((8,)),
        hypothesis((9,), "decreases"),
        hypothesis((10,), "increases"),
        hypothesis((11,), "decreases"),
        "IF x999 = 1 THEN y = 1 increases",
        "x0 causes y",
    ]
    null_features = list(range(9, 14))
    for _ in range(n_random_null):
        size = int(rng.integers(1, 3))
        feats = tuple(sorted(rng.choice(null_features, size=size, replace=False).tolist()))
        direction = "increases" if rng.random() < 0.5 else "decreases"
        proposals.append(hypothesis(feats, direction))
    return proposals


def exhaustive_proposals(n_features: int, max_len: int = 2) -> list[str]:
    proposals = []
    for size in range(1, max_len + 1):
        for feats in combinations(range(n_features), size):
            proposals.append(hypothesis(tuple(feats), "increases"))
            proposals.append(hypothesis(tuple(feats), "decreases"))
    return proposals


def category(text: str) -> str:
    if "x999" in text or "causes" in text:
        return "uncompiled"
    if "x0 = 1 AND x1 = 1" in text or "x0 = 1 THEN" in text or "x1 = 1 THEN" in text:
        return "true"
    if "x2 = 1" in text or "x3 = 1" in text:
        return "shifted"
    if "x4 = 1" in text:
        return "rare"
    if "x5 = 1" in text or "x6 = 1" in text:
        return "redundant"
    if "x7 = 1" in text or "x8 = 1" in text:
        return "confounded"
    return "null"


def summarize_cards(cards, proposals: list[str]) -> dict[str, object]:
    accepted = [card for card in cards if card.decision == "accepted"]
    rejected = [card for card in cards if card.decision != "accepted"]
    accepted_categories = [category(card.hypothesis) for card in accepted]
    rejected_reasons: dict[str, int] = {}
    for card in rejected:
        for reason in card.reason.split("; "):
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
    return {
        "proposed": len(proposals),
        "unique": len(set(proposals)),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_true": accepted_categories.count("true"),
        "accepted_shifted": accepted_categories.count("shifted"),
        "accepted_rare": accepted_categories.count("rare"),
        "accepted_redundant": accepted_categories.count("redundant"),
        "accepted_confounded": accepted_categories.count("confounded"),
        "accepted_null": accepted_categories.count("null"),
        "rejected_support": rejected_reasons.get("insufficient support", 0),
        "rejected_fdr": sum(v for k, v in rejected_reasons.items() if "not significant" in k),
        "rejected_counterfactual": rejected_reasons.get("matched counterfactual contrast is weak", 0),
        "rejected_holdout": rejected_reasons.get("held-out effect does not replicate", 0),
    }


def main() -> None:
    rows = []
    reason_rows = []
    for seed in range(8):
        data = make_stress_data(n_per_split=900, seed=seed)
        rng = np.random.default_rng(seed + 123)
        suites = {
            "LLMStress": stress_proposals(rng),
            "ExhaustiveRules": exhaustive_proposals(10, max_len=2),
        }
        for suite, proposals in suites.items():
            for method, kwargs in {
                "LLMOnly": {
                    "use_support": False,
                    "use_statistical_test": False,
                    "use_fdr": False,
                    "use_counterfactual": False,
                    "use_holdout": False,
                    "min_effect": -1.0,
                },
                "StatisticalFilter": {
                    "use_support": True,
                    "use_statistical_test": True,
                    "use_fdr": True,
                    "use_counterfactual": False,
                    "use_holdout": False,
                },
                "FullVerifier": {},
            }.items():
                cards = verify_hypotheses(
                    data,
                    proposals,
                    permutation_rounds=60,
                    bootstrap_rounds=60,
                    **kwargs,
                )
                summary = summarize_cards(cards, proposals)
                rows.append({"seed": seed, "suite": suite, "method": method, **summary})
                for card in cards:
                    if card.decision != "accepted":
                        reason_rows.append(
                            {
                                "seed": seed,
                                "suite": suite,
                                "method": method,
                                "category": category(card.hypothesis),
                                "reason": card.reason,
                            }
                        )

    metrics = [k for k in rows[0] if k not in {"seed", "suite", "method"}]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["suite"]), str(row["method"])), []).append(row)
    summary_rows = []
    for (suite, method), group in sorted(grouped.items()):
        out: dict[str, object] = {"suite": suite, "method": method, "runs": len(group)}
        for metric in metrics:
            values = np.array([float(row[metric]) for row in group])
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std())
        summary_rows.append(out)

    result_dir = ROOT / "results"
    write_csv(result_dir / "verifiable_stress_runs.csv", rows)
    write_csv(result_dir / "verifiable_stress_summary.csv", summary_rows)
    write_csv(result_dir / "verifiable_stress_rejections.csv", reason_rows)
    print(f"Wrote {result_dir / 'verifiable_stress_summary.csv'}")
    for row in summary_rows:
        print(
            f"{row['suite']:15s} {row['method']:17s} "
            f"accepted={float(row['accepted_mean']):.1f} "
            f"shifted={float(row['accepted_shifted_mean']):.1f} "
            f"null={float(row['accepted_null_mean']):.1f}"
        )


if __name__ == "__main__":
    main()
