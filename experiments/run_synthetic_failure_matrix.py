"""Synthetic failure-mode matrix and null-scale FDR sweep.

This script is intentionally more diagnostic than the original shifted-rule
benchmark. It creates one regime per verifier gate and records which gate fires.
It also runs a log-scale multiple-testing sweep with simulated null proposal
p-values to show raw-testing FDR growth and BH control.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_gate_necessity import CONFIGS, make_gate_data  # noqa: E402
from verifiable_discovery import verify_hypotheses, benjamini_hochberg  # noqa: E402


RESULTS = ROOT / "results"
REGIMES = {
    "A_heldout": "holdout",
    "B_fdr": "fdr_null",
    "C_support": "support",
    "D_counterfactual": "counterfactual",
    "E_permutation": "statistical",
}
METHODS = ["FullVerifier", "NoSupport", "NoStatTest", "NoFDR", "NoCounterfactual", "NoHeldout"]
NULL_COUNTS = [10, 30, 100, 300, 1000]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    mean = float(arr.mean()) if len(arr) else 0.0
    if len(arr) <= 1:
        return mean, mean, mean
    half = 1.96 * float(arr.std(ddof=1)) / math.sqrt(len(arr))
    return mean, mean - half, mean + half


def run_matrix(seeds: int = 50) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for seed in range(seeds):
        data, proposals, labels = make_gate_data(seed, n_per_split=650)
        for regime, failure_type in REGIMES.items():
            regime_props = [p for p, t in labels.items() if t in {"valid", failure_type}]
            for method in METHODS:
                config = {"min_support": 0.05, "min_effect": 0.045, "alpha": 0.05, **CONFIGS.get(method, {})}
                cards = verify_hypotheses(
                    data,
                    regime_props,
                    permutation_rounds=120,
                    bootstrap_rounds=120,
                    **config,
                )
                for card in cards:
                    failure = labels.get(card.hypothesis, "unknown")
                    rows.append(
                        {
                            "seed": seed,
                            "regime": regime,
                            "target_failure": failure_type,
                            "method": method,
                            "failure_type": failure,
                            "decision": card.decision,
                            "reason": card.reason,
                            "accepted_invalid": int(card.decision == "accepted" and failure == failure_type),
                            "support_gate": int("insufficient support" in card.reason),
                            "effect_gate": int("weak validation effect" in card.reason),
                            "stat_fdr_gate": int("not significant" in card.reason),
                            "counterfactual_gate": int("counterfactual" in card.reason),
                            "heldout_gate": int("held-out" in card.reason),
                        }
                    )

    summary: list[dict[str, object]] = []
    for regime in REGIMES:
        for method in METHODS:
            subset = [r for r in rows if r["regime"] == regime and r["method"] == method and r["failure_type"] == r["target_failure"]]
            by_seed = []
            for seed in range(seeds):
                seed_rows = [r for r in subset if r["seed"] == seed]
                by_seed.append(float(any(r["accepted_invalid"] for r in seed_rows)))
            mean, low, high = ci95(by_seed)
            summary.append(
                {
                    "regime": regime,
                    "method": method,
                    "invalid_accept_rate": mean,
                    "invalid_accept_ci_low": low,
                    "invalid_accept_ci_high": high,
                    "seeds": seeds,
                    "support_gate_fires": sum(int(r["support_gate"]) for r in subset),
                    "effect_gate_fires": sum(int(r["effect_gate"]) for r in subset),
                    "stat_fdr_gate_fires": sum(int(r["stat_fdr_gate"]) for r in subset),
                    "counterfactual_gate_fires": sum(int(r["counterfactual_gate"]) for r in subset),
                    "heldout_gate_fires": sum(int(r["heldout_gate"]) for r in subset),
                }
            )
    return rows, summary


def run_null_sweep(seeds: int = 200) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    alpha = 0.05
    true_p = [0.001, 0.003]
    for null_count in NULL_COUNTS:
        harmonic = sum(1.0 / k for k in range(1, null_count + len(true_p) + 1))
        for seed in range(seeds):
            rng = np.random.default_rng(10000 + null_count * 17 + seed)
            null_p = rng.random(null_count).tolist()
            pvals = true_p + null_p
            raw_accept = [p <= alpha for p in pvals]
            bh_q = benjamini_hochberg(pvals)
            bh_accept = [q <= alpha for q in bh_q]
            by_accept = [q <= alpha / harmonic for q in bh_q]
            for method, accepted in [("RawP", raw_accept), ("BH", bh_accept), ("BY", by_accept)]:
                null_accepts = sum(accepted[2:])
                total_accepts = sum(accepted)
                rows.append(
                    {
                        "seed": seed,
                        "null_proposals": null_count,
                        "method": method,
                        "accepted": total_accepts,
                        "null_accepted": null_accepts,
                        "achieved_fdr": null_accepts / total_accepts if total_accepts else 0.0,
                    }
                )
    return rows


def summarize_null_sweep(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for null_count in NULL_COUNTS:
        for method in ["RawP", "BH", "BY"]:
            vals = [float(r["achieved_fdr"]) for r in rows if r["null_proposals"] == null_count and r["method"] == method]
            mean, low, high = ci95(vals)
            out.append(
                {
                    "null_proposals": null_count,
                    "method": method,
                    "achieved_fdr_mean": mean,
                    "achieved_fdr_ci_low": low,
                    "achieved_fdr_ci_high": high,
                    "seeds": len(vals),
                }
            )
    return out


def main() -> None:
    matrix_rows, matrix_summary = run_matrix()
    sweep_rows = run_null_sweep()
    sweep_summary = summarize_null_sweep(sweep_rows)
    write_csv(RESULTS / "synthetic_failure_matrix_runs.csv", matrix_rows)
    write_csv(RESULTS / "synthetic_failure_matrix_summary.csv", matrix_summary)
    write_csv(RESULTS / "null_fdr_sweep_runs.csv", sweep_rows)
    write_csv(RESULTS / "null_fdr_sweep_summary.csv", sweep_summary)
    print(f"Wrote {RESULTS / 'synthetic_failure_matrix_summary.csv'}")
    print(f"Wrote {RESULTS / 'null_fdr_sweep_summary.csv'}")
    for row in sweep_summary:
        print(f"{row['method']:4s} nulls={int(row['null_proposals']):4d} fdr={row['achieved_fdr_mean']:.3f}")


if __name__ == "__main__":
    main()
