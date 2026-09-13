"""Gate-necessity benchmark for VeraDM.

The earlier synthetic benchmark isolated held-out shift. This benchmark creates
several failure modes in one controlled dataset so that every verifier gate is
necessary for at least one proposed hypothesis:

- support: rare high-effect rule;
- statistical test: small noisy apparent effect;
- FDR: many null proposals with one validation false positive;
- counterfactual: marginal association explained by an observed context;
- held-out: validation association reverses in the held-out split.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verifiable_discovery import Dataset, EvidenceCard, verify_hypotheses  # noqa: E402


RESULTS = ROOT / "results"


CONFIGS = {
    "FullVerifier": {},
    "NoSupport": {"use_support": False},
    "NoStatTest": {"use_statistical_test": False},
    "NoFDR": {"use_fdr": False},
    "NoCounterfactual": {"use_counterfactual": False},
    "NoHeldout": {"use_holdout": False},
    "NoEffect": {"min_effect": -1.0},
}


PROPOSAL_INFO = [
    ("true_stable", "IF x0 = 1 THEN y = 1 increases", "valid"),
    ("rare_high_effect", "IF x2 = 1 THEN y = 1 increases", "support"),
    ("noisy_effect", "IF x3 = 1 THEN y = 1 increases", "statistical"),
    ("confounded", "IF x4 = 1 THEN y = 1 increases", "counterfactual"),
    ("shifted", "IF x5 = 1 THEN y = 1 increases", "holdout"),
    ("redundant_alias", "IF x6 = 1 THEN y = 1 increases", "redundant"),
]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_gate_data(seed: int, n_per_split: int = 900) -> tuple[Dataset, list[str], dict[str, str]]:
    rng = np.random.default_rng(seed)
    xs, ys, splits = [], [], []
    for split in (0, 1, 2):
        x = np.zeros((n_per_split, 28), dtype=int)
        context = rng.binomial(1, 0.5, n_per_split)
        x[:, 1] = context
        x[:, 0] = rng.binomial(1, 0.5, n_per_split)

        logit = -2.0 + 1.0 * x[:, 0] + 1.4 * context
        y = rng.binomial(1, sigmoid(logit))

        # Rare high-effect rule: strong and replicated, but below support.
        rare = rng.binomial(1, 0.022, n_per_split)
        y = np.where(rare == 1, rng.binomial(1, 0.95, n_per_split), y)
        x[:, 2] = rare

        # Noisy effect: just enough validation support and apparent direction,
        # but too little validation evidence for the permutation test. The same
        # rule is stronger in holdout so that the statistical gate, not held-out
        # replication, is the distinctive blocker.
        noisy_support = 0.055 if split == 1 else 0.24
        noisy = rng.binomial(1, noisy_support, n_per_split)
        if split == 1:
            y = np.where(noisy == 1, rng.binomial(1, 0.30, n_per_split), y)
        if split == 2:
            y = np.where(noisy == 1, rng.binomial(1, 0.55, n_per_split), y)
        x[:, 3] = noisy

        # Confounded rule: proxy for context, stable marginally but weak inside context strata.
        x[:, 4] = context

        # Shifted rule: validation looks strong, hold-out reverses.
        if split in (0, 1):
            x[:, 5] = np.where(y == 1, rng.binomial(1, 0.82, n_per_split), rng.binomial(1, 0.22, n_per_split))
        else:
            x[:, 5] = np.where(y == 1, rng.binomial(1, 0.18, n_per_split), rng.binomial(1, 0.80, n_per_split))

        # Redundant alias of the true stable rule.
        x[:, 6] = np.where(x[:, 0] == 1, rng.binomial(1, 0.93, n_per_split), rng.binomial(1, 0.07, n_per_split))

        # Null bank for FDR pressure. On validation, seed-specific sampling can
        # produce raw p<alpha false positives; BH should remove them.
        x[:, 7:] = rng.binomial(1, 0.5, size=(n_per_split, x.shape[1] - 7))

        xs.append(x)
        ys.append(y)
        splits.append(np.full(n_per_split, split))

    data = Dataset(np.vstack(xs), np.concatenate(ys), np.concatenate(splits), tuple(f"x{i}" for i in range(28)))
    proposals = [item[1] for item in PROPOSAL_INFO]
    labels = {item[1]: item[2] for item in PROPOSAL_INFO}
    for feature in range(7, 28):
        text = f"IF x{feature} = 1 THEN y = 1 increases"
        proposals.append(text)
        labels[text] = "fdr_null"
    return data, proposals, labels


def rows_for_cards(seed: int, method: str, cards: list[EvidenceCard], labels: dict[str, str]) -> list[dict[str, object]]:
    rows = []
    for card in cards:
        rows.append(
            {
                "seed": seed,
                "method": method,
                "hypothesis": card.hypothesis,
                "failure_type": labels.get(card.hypothesis, "unknown"),
                "decision": card.decision,
                "support_validation": card.support_validation,
                "support_holdout": card.support_holdout,
                "validation_effect": card.validation_effect,
                "counterfactual_effect": card.counterfactual_effect,
                "holdout_effect": card.holdout_effect,
                "p_value": card.p_value,
                "q_value": card.q_value,
                "reason": card.reason,
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for method in sorted({str(r["method"]) for r in rows}):
        subset = [r for r in rows if r["method"] == method and r["decision"] == "accepted"]
        accepted_types = {str(r["failure_type"]) for r in subset}
        for failure_type in ["valid", "support", "statistical", "fdr_null", "counterfactual", "holdout", "redundant"]:
            out.append(
                {
                    "method": method,
                    "failure_type": failure_type,
                    "accepted": sum(1 for r in subset if r["failure_type"] == failure_type),
                    "seeds": len({r["seed"] for r in rows if r["method"] == method}),
                    "accepted_on_any_seed": int(failure_type in accepted_types),
                }
            )
    return out


def main() -> None:
    rows: list[dict[str, object]] = []
    for seed in range(10):
        data, proposals, labels = make_gate_data(seed)
        for method, kwargs in CONFIGS.items():
            config = {
                "min_support": 0.05,
                "min_effect": 0.045,
                "alpha": 0.05,
                **kwargs,
            }
            cards = verify_hypotheses(
                data,
                proposals,
                permutation_rounds=250,
                bootstrap_rounds=250,
                **config,
            )
            rows.extend(rows_for_cards(seed, method, cards, labels))

    summary_rows = summarize(rows)
    write_csv(RESULTS / "gate_necessity_runs.csv", rows)
    write_csv(RESULTS / "gate_necessity_summary.csv", summary_rows)
    print(f"Wrote {RESULTS / 'gate_necessity_summary.csv'}")
    for row in summary_rows:
        if row["accepted_on_any_seed"]:
            print(f"{row['method']:18s} accepts {row['failure_type']}: {row['accepted']}")


if __name__ == "__main__":
    main()
