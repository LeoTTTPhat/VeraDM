"""Weighted-BH confidence diagnostic on a real temporal case.

Theorem 2 says that LLM confidence can be converted into pre-held-out weights
without losing FDR control when the weights are independent of the final test
split. This diagnostic asks whether the weighting is useful empirically.

We use Bank Marketing evidence cards because the candidate set contains both
stable held-out-replicating rules and validation-significant proxy rules that do
not replicate. Since real data do not expose ground-truth hypotheses, the
positive class is operational: a rule is "replicating" if it has held-out support
and a held-out effect in the stated direction. We then compare uniform BH with
weighted BH under calibrated and deliberately miscalibrated confidence scores.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
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


def bh_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    thresholds = alpha * np.arange(1, m + 1) / m
    passed = p_values[order] <= thresholds
    rejected = np.zeros(m, dtype=bool)
    if passed.any():
        cutoff = np.where(passed)[0].max()
        rejected[order[: cutoff + 1]] = True
    return rejected


def normalized_weights(confidence: np.ndarray) -> np.ndarray:
    raw = np.exp(2.0 * (confidence - 0.5))
    return raw / raw.mean()


def evaluate(
    p_values: np.ndarray,
    replicates: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    rejected = bh_rejections(p_values / weights, alpha)
    discoveries = int(rejected.sum())
    true_discoveries = int((rejected & replicates).sum())
    false_discoveries = int((rejected & ~replicates).sum())
    return {
        "alpha": alpha,
        "discoveries": discoveries,
        "replicating_discoveries": true_discoveries,
        "false_discoveries": false_discoveries,
        "empirical_fdr": false_discoveries / max(discoveries, 1),
        "power": true_discoveries / max(int(replicates.sum()), 1),
    }


def main() -> None:
    frame = pd.read_csv(RESULTS / "bank_temporal_evidence_cards.csv")
    direction = np.where(frame["hypothesis"].astype(str).str.contains("decreases"), -1.0, 1.0)
    min_support = 0.025
    min_effect = 0.035
    replicates = (
        (frame["support_holdout"].to_numpy() >= min_support)
        & (direction * frame["holdout_effect"].to_numpy() >= min_effect)
    )
    p_values = frame["p_value"].to_numpy(dtype=float)

    confidence = {
        "Uniform BH": np.full(len(frame), 0.5),
        "Weighted BH, calibrated": np.where(replicates, 0.90, 0.10),
        "Weighted BH, miscalibrated": np.where(replicates, 0.10, 0.90),
    }
    alpha_grid = np.array([0.002, 0.003, 0.005, 0.010, 0.020, 0.050, 0.100])
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for method, scores in confidence.items():
        weights = normalized_weights(scores)
        for alpha in alpha_grid:
            metrics = evaluate(p_values, replicates, weights, float(alpha))
            curve_rows.append(
                {
                    "dataset": "Bank temporal",
                    "method": method,
                    "mean_weight_replicating": float(weights[replicates].mean()),
                    "mean_weight_nonreplicating": float(weights[~replicates].mean()),
                    **metrics,
                }
            )
        eligible = [row for row in curve_rows if row["method"] == method and row["empirical_fdr"] <= 0.10]
        best = max(eligible, key=lambda row: (row["power"], row["alpha"]))
        summary_rows.append(
            {
                "dataset": "Bank temporal",
                "method": method,
                "selected_alpha": best["alpha"],
                "discoveries": best["discoveries"],
                "replicating_discoveries": best["replicating_discoveries"],
                "false_discoveries": best["false_discoveries"],
                "empirical_fdr": best["empirical_fdr"],
                "power": best["power"],
                "mean_weight_replicating": float(weights[replicates].mean()),
                "mean_weight_nonreplicating": float(weights[~replicates].mean()),
            }
        )

    calibrated_alpha = next(
        row["selected_alpha"] for row in summary_rows if row["method"] == "Weighted BH, calibrated"
    )
    mis_weights = normalized_weights(confidence["Weighted BH, miscalibrated"])
    mis_same_alpha = evaluate(p_values, replicates, mis_weights, float(calibrated_alpha))
    summary_rows.append(
        {
            "dataset": "Bank temporal",
            "method": "Weighted BH, miscalibrated at calibrated alpha",
            "selected_alpha": mis_same_alpha["alpha"],
            "discoveries": mis_same_alpha["discoveries"],
            "replicating_discoveries": mis_same_alpha["replicating_discoveries"],
            "false_discoveries": mis_same_alpha["false_discoveries"],
            "empirical_fdr": mis_same_alpha["empirical_fdr"],
            "power": mis_same_alpha["power"],
            "mean_weight_replicating": float(mis_weights[replicates].mean()),
            "mean_weight_nonreplicating": float(mis_weights[~replicates].mean()),
        }
    )

    write_csv(RESULTS / "weighted_bh_confidence_curve.csv", curve_rows)
    write_csv(RESULTS / "weighted_bh_confidence_summary.csv", summary_rows)
    print("Weighted-BH diagnostic complete.")
    print(f"Replicating proxy positives={int(replicates.sum())}/{len(replicates)}")
    for row in summary_rows:
        print(
            f"{row['method']}: alpha={row['selected_alpha']:.3f}, "
            f"discoveries={row['discoveries']}, FDR={row['empirical_fdr']:.2f}, "
            f"power={row['power']:.2f}"
        )


if __name__ == "__main__":
    main()
