"""Semi-synthetic injection benchmark on real tabular covariates."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_verifiable_discovery import load_adult_income, load_german_credit  # noqa: E402
from verifiable_discovery import Dataset, verify_hypotheses  # noqa: E402


RESULTS = ROOT / "results"
BANK_DATA = ROOT / "data" / "real" / "bank_month_env.csv"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def make_injected_dataset(base: Dataset, seed: int) -> tuple[Dataset, list[str], dict[str, str]]:
    rng = np.random.default_rng(seed)
    x = base.x.astype(int)
    n_features = x.shape[1]
    true_rules = [(0,), (1,), (0, 1)]
    null_features = [f for f in range(min(n_features, 12)) if f not in {0, 1}]
    proposals: list[str] = []
    labels: dict[str, str] = {}

    logit = -2.0 + 1.1 * x[:, 0] + 0.9 * x[:, 1] + 1.0 * (x[:, 0] & x[:, 1])
    y = rng.binomial(1, sigmoid(logit))
    injected = Dataset(x=x, y=y, split=base.split, feature_names=base.feature_names)

    for rule in true_rules:
        antecedent = " AND ".join(f"x{i} = 1" for i in rule)
        text = f"IF {antecedent} THEN y = 1 increases"
        proposals.append(text)
        labels[text] = "injected_true"

    for feature in null_features[:12]:
        text = f"IF x{feature} = 1 THEN y = 1 increases"
        if text in labels:
            continue
        proposals.append(text)
        labels[text] = "known_null"
    for i, feature_i in enumerate(null_features[:6]):
        for feature_j in null_features[i + 1 : i + 3]:
            text = f"IF x{feature_i} = 1 AND x{feature_j} = 1 THEN y = 1 increases"
            proposals.append(text)
            labels[text] = "known_null"
    return injected, proposals, labels


def load_bank_temporal_injection() -> Dataset:
    if not BANK_DATA.exists():
        raise FileNotFoundError(
            f"Bank Marketing data not found at {BANK_DATA}.\n"
            "Run: python prepare_data.py --bank"
        )
    frame = pd.read_csv(BANK_DATA)
    features = pd.DataFrame(
        {
            "bank_age_ge_50": frame["age"] >= 50,
            "bank_job_retired": frame["job"].eq("retired"),
            "bank_university_degree": frame["education"].eq("university.degree"),
            "bank_default_unknown": frame["default"].eq("unknown"),
            "bank_housing_yes": frame["housing"].eq("yes"),
            "bank_loan_yes": frame["loan"].eq("yes"),
            "bank_contact_cellular": frame["contact"].eq("cellular"),
            "bank_duration_ge_300": frame["duration"] >= 300,
            "bank_campaign_gt_2": frame["campaign"] > 2,
            "bank_previous_contact": frame["previous"] > 0,
            "bank_poutcome_success": frame["poutcome"].eq("success"),
            "bank_euribor_low": frame["euribor3m"] < 2.0,
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


def metrics(cards, labels: dict[str, str]) -> dict[str, object]:
    accepted = [card for card in cards if card.decision == "accepted"]
    true_total = sum(1 for label in labels.values() if label == "injected_true")
    true_accepts = sum(1 for card in accepted if labels.get(card.hypothesis) == "injected_true")
    null_accepts = sum(1 for card in accepted if labels.get(card.hypothesis) == "known_null")
    return {
        "accepted": len(accepted),
        "true_accepts": true_accepts,
        "null_accepts": null_accepts,
        "recall": true_accepts / true_total if true_total else 0.0,
        "empirical_fdr": null_accepts / len(accepted) if accepted else 0.0,
    }


def ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean, mean, mean
    half = 1.96 * float(arr.std(ddof=1)) / np.sqrt(len(arr))
    return mean, mean - half, mean + half


def main() -> None:
    loaders = {
        "Adult": load_adult_income,
        "German": load_german_credit,
        "BankTemporal": load_bank_temporal_injection,
    }
    rows = []
    for dataset, loader in loaders.items():
        base = loader()
        for seed in range(30):
            injected, proposals, labels = make_injected_dataset(base, seed)
            cards = verify_hypotheses(
                injected,
                proposals,
                min_support=0.04,
                min_effect=0.04,
                alpha=0.05,
                permutation_rounds=140,
                bootstrap_rounds=140,
            )
            rows.append({"dataset": dataset, "seed": seed, "proposals": len(proposals), **metrics(cards, labels)})

    summary = []
    for dataset in loaders:
        subset = [r for r in rows if r["dataset"] == dataset]
        out: dict[str, object] = {"dataset": dataset, "runs": len(subset)}
        for key in ["accepted", "recall", "empirical_fdr", "true_accepts", "null_accepts"]:
            mean, low, high = ci95([float(r[key]) for r in subset])
            out[f"{key}_mean"] = mean
            out[f"{key}_ci_low"] = low
            out[f"{key}_ci_high"] = high
        summary.append(out)

    write_csv(RESULTS / "real_injection_runs.csv", rows)
    write_csv(RESULTS / "real_injection_summary.csv", summary)
    print(f"Wrote {RESULTS / 'real_injection_summary.csv'}")
    for row in summary:
        print(f"{row['dataset']:12s} recall={row['recall_mean']:.2f} fdr={row['empirical_fdr_mean']:.2f}")


if __name__ == "__main__":
    main()
