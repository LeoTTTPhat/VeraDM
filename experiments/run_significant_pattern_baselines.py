"""Published-style significant-pattern-mining baselines.

The goal is not to reproduce a particular implementation package, but to add
recognizable statistical pattern-mining controls:

- BH-Exhaustive: exhaustive singleton/pair rules with BH correction;
- Tarone-Bonferroni: testability-style support-dependent Bonferroni;
- Westfall-Young: maxT permutation correction over the enumerated rule family;
- VeraDM: same rule family with counterfactual and held-out gates.
"""

from __future__ import annotations

import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verifiable_discovery import (  # noqa: E402
    Dataset,
    benjamini_hochberg,
    compile_hypothesis,
    discovery_split,
    make_discovery_data,
    matched_strata_effect,
    pattern_indicator,
    permutation_p_value,
    risk_difference,
    support,
    verify_hypotheses,
)


RESULTS = ROOT / "results"
TRUE_PATTERNS = {("x0",), ("x1",), ("x0", "x1")}
SHIFTED = {("x2",)}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def proposal(features: tuple[int, ...], direction: str = "increases") -> str:
    antecedent = " AND ".join(f"x{feature} = 1" for feature in features)
    return f"IF {antecedent} THEN y = 1 {direction}"


def enumerate_rules(n_features: int = 8, max_len: int = 2) -> list[str]:
    out = []
    for size in range(1, max_len + 1):
        for feats in combinations(range(n_features), size):
            out.append(proposal(feats, "increases"))
            out.append(proposal(feats, "decreases"))
    return out


def pattern_key(text: str) -> tuple[str, ...]:
    hyp = compile_hypothesis(text, tuple(f"x{i}" for i in range(8)))
    return tuple(f"x{p.feature}" for p in hyp.predicates)


def signed_measurements(data: Dataset, proposals: list[str]) -> list[dict[str, object]]:
    vx, vy = discovery_split(data, 1)
    hx, hy = discovery_split(data, 2)
    rows = []
    for text in proposals:
        hyp = compile_hypothesis(text, data.feature_names)
        active_val = pattern_indicator(vx, hyp.predicates)
        active_hold = pattern_indicator(hx, hyp.predicates)
        effect = risk_difference(vy, active_val)
        hold = risk_difference(hy, active_hold)
        p_value = permutation_p_value(vy, active_val, hyp.direction, rounds=180)
        rows.append(
            {
                "hypothesis": text,
                "key": pattern_key(text),
                "direction": hyp.direction,
                "support_validation": support(active_val),
                "support_holdout": support(active_hold),
                "validation_effect": effect,
                "signed_validation": hyp.direction * effect,
                "holdout_effect": hold,
                "signed_holdout": hyp.direction * hold,
                "counterfactual": hyp.direction * matched_strata_effect(vx, vy, hyp),
                "p_value": p_value,
            }
        )
    return rows


def westfall_young_threshold(data: Dataset, proposals: list[str], *, alpha: float, rounds: int = 120) -> float:
    vx, vy = discovery_split(data, 1)
    compiled = [compile_hypothesis(text, data.feature_names) for text in proposals]
    actives = [pattern_indicator(vx, hyp.predicates) for hyp in compiled]
    rng = np.random.default_rng(917)
    max_stats = []
    for _ in range(rounds):
        shuffled = rng.permutation(vy)
        stats = []
        for hyp, active in zip(compiled, actives):
            stats.append(hyp.direction * risk_difference(shuffled, active))
        max_stats.append(max(stats))
    return float(np.quantile(max_stats, 1.0 - alpha))


def run_baseline(data: Dataset, proposals: list[str], method: str) -> list[str]:
    alpha = 0.05
    min_support = 0.05
    min_effect = 0.05
    rows = signed_measurements(data, proposals)
    if method == "BH-Exhaustive":
        q_values = benjamini_hochberg([float(r["p_value"]) for r in rows])
        return [
            str(r["hypothesis"])
            for r, q in zip(rows, q_values)
            if r["support_validation"] >= min_support and r["signed_validation"] >= min_effect and q <= alpha
        ]
    if method == "Tarone-Bonferroni":
        testable = [r for r in rows if r["support_validation"] >= min_support]
        m_testable = max(1, len(testable))
        return [
            str(r["hypothesis"])
            for r in testable
            if r["signed_validation"] >= min_effect and float(r["p_value"]) <= alpha / m_testable
        ]
    if method == "Westfall-Young":
        threshold = westfall_young_threshold(data, proposals, alpha=alpha)
        return [
            str(r["hypothesis"])
            for r in rows
            if r["support_validation"] >= min_support and r["signed_validation"] >= max(min_effect, threshold)
        ]
    if method == "Subgroup-BH":
        candidates = [
            r
            for r in rows
            if r["support_validation"] >= min_support and r["signed_validation"] >= min_effect and len(r["key"]) <= 2
        ]
        candidates = sorted(candidates, key=lambda r: float(r["signed_validation"]), reverse=True)[:30]
        q_values = benjamini_hochberg([float(r["p_value"]) for r in candidates])
        return [str(r["hypothesis"]) for r, q in zip(candidates, q_values) if q <= alpha]
    if method == "StableEnv-BH":
        q_values = benjamini_hochberg([float(r["p_value"]) for r in rows])
        return [
            str(r["hypothesis"])
            for r, q in zip(rows, q_values)
            if r["support_validation"] >= min_support
            and r["support_holdout"] >= min_support
            and r["signed_validation"] >= min_effect
            and r["signed_holdout"] >= min_effect
            and np.sign(float(r["validation_effect"])) == np.sign(float(r["holdout_effect"]))
            and q <= alpha
        ]
    if method == "VeraDM":
        cards = verify_hypotheses(
            data,
            proposals,
            min_support=min_support,
            min_effect=min_effect,
            alpha=alpha,
            permutation_rounds=180,
            bootstrap_rounds=180,
        )
        return [card.hypothesis for card in cards if card.decision == "accepted"]
    raise ValueError(method)


def metrics(accepted: list[str]) -> dict[str, object]:
    keys = {pattern_key(text) for text in accepted}
    true_hits = len(keys & TRUE_PATTERNS)
    shifted = len(keys & SHIFTED)
    false = len(keys - TRUE_PATTERNS)
    signal_proxy = {key for key in keys if {"x0", "x1"}.intersection(key)}
    proxy_false = len(keys - signal_proxy - SHIFTED)
    proxy_aware_false = len(keys - signal_proxy)
    return {
        "accepted": len(keys),
        "true_hits": true_hits,
        "recall": true_hits / len(TRUE_PATTERNS),
        "false_discoveries": false,
        "empirical_fdr": false / len(keys) if keys else 0.0,
        "proxy_aware_false_discoveries": proxy_aware_false,
        "proxy_aware_fdr": proxy_aware_false / len(keys) if keys else 0.0,
        "null_only_false_discoveries": proxy_false,
        "signal_proxy_rules": len(signal_proxy),
        "shifted_accepts": shifted,
    }


def main() -> None:
    proposals = enumerate_rules(8, max_len=2)
    methods = ["BH-Exhaustive", "Tarone-Bonferroni", "Westfall-Young", "Subgroup-BH", "StableEnv-BH", "VeraDM"]
    rows = []
    for seed in range(10):
        data = make_discovery_data(n_per_split=900, seed=seed)
        for method in methods:
            accepted = run_baseline(data, proposals, method)
            rows.append({"seed": seed, "method": method, **metrics(accepted)})

    summary = []
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        out: dict[str, object] = {"method": method, "runs": len(subset)}
        for metric in [
            "accepted",
            "true_hits",
            "recall",
            "false_discoveries",
            "empirical_fdr",
            "proxy_aware_false_discoveries",
            "proxy_aware_fdr",
            "null_only_false_discoveries",
            "signal_proxy_rules",
            "shifted_accepts",
        ]:
            values = np.array([float(r[metric]) for r in subset])
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std())
        summary.append(out)

    write_csv(RESULTS / "significant_pattern_baselines_runs.csv", rows)
    write_csv(RESULTS / "significant_pattern_baselines_summary.csv", summary)

    easy_rows = []
    easy_data = make_discovery_data(n_per_split=2500, seed=999)
    easy_props = [
        "IF x0 = 1 THEN y = 1 increases",
        "IF x1 = 1 THEN y = 1 increases",
        "IF x0 = 1 AND x1 = 1 THEN y = 1 increases",
    ]
    for method in ["Tarone-Bonferroni", "BH-Exhaustive"]:
        accepted = run_baseline(easy_data, easy_props, method)
        easy_rows.append({"method": method, "candidate_rules": len(easy_props), **metrics(accepted)})
    write_csv(RESULTS / "tarone_sanity_check.csv", easy_rows)
    print(f"Wrote {RESULTS / 'significant_pattern_baselines_summary.csv'}")
    for row in summary:
        print(
            f"{row['method']:20s} accepted={float(row['accepted_mean']):.1f} "
            f"recall={float(row['recall_mean']):.2f} fdr={float(row['empirical_fdr_mean']):.2f} "
            f"shifted={float(row['shifted_accepts_mean']):.1f}"
        )


if __name__ == "__main__":
    main()
