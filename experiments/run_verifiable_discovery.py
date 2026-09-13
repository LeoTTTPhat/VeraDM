"""Run LLM-guided verifiable discovery experiments.

The script conducts the planned study:
- synthetic benchmark with known true, spurious, and null discoveries;
- baseline comparison against LLM-only and weaker verification variants;
- ablation study over individual verifier gates;
- two public UCI tabular datasets with fallback synthetic replacements.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real"
sys.path.insert(0, str(ROOT / "src"))

from verifiable_discovery import (  # noqa: E402
    Dataset,
    EvidenceCard,
    compile_hypothesis,
    make_discovery_data,
    prune_redundant_cards,
    verify_hypotheses,
)
from llm_proposals import (  # noqa: E402
    OpenAIHypothesisProvider,
    ProposalBatch,
    proposal_diagnostics,
    save_batch,
    top_association_proposals,
)


TRUE_PATTERNS = {(0,), (1,), (0, 1)}
SPURIOUS_PATTERNS = {(2,)}


BASELINE_CONFIGS = {
    "LLMOnly": {
        "use_support": False,
        "use_statistical_test": False,
        "use_fdr": False,
        "use_counterfactual": False,
        "use_holdout": False,
        "min_effect": -1.0,
    },
    "DescriptiveStats": {
        "use_support": True,
        "use_statistical_test": False,
        "use_fdr": False,
        "use_counterfactual": False,
        "use_holdout": False,
    },
    "StatisticalFilter": {
        "use_support": True,
        "use_statistical_test": True,
        "use_fdr": True,
        "use_counterfactual": False,
        "use_holdout": False,
    },
    "HeldoutVerifier": {
        "use_support": True,
        "use_statistical_test": True,
        "use_fdr": True,
        "use_counterfactual": False,
        "use_holdout": True,
    },
    "FullVerifier": {
        "use_support": True,
        "use_statistical_test": True,
        "use_fdr": True,
        "use_counterfactual": True,
        "use_holdout": True,
    },
}


ABLATION_CONFIGS = {
    "FullVerifier": BASELINE_CONFIGS["FullVerifier"],
    "NoSupportGate": {**BASELINE_CONFIGS["FullVerifier"], "use_support": False},
    "NoStatisticalTest": {**BASELINE_CONFIGS["FullVerifier"], "use_statistical_test": False},
    "NoFDRCorrection": {**BASELINE_CONFIGS["FullVerifier"], "use_fdr": False},
    "NoCounterfactual": {**BASELINE_CONFIGS["FullVerifier"], "use_counterfactual": False},
    "NoHeldout": {**BASELINE_CONFIGS["FullVerifier"], "use_holdout": False},
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pattern_key_from_text(hypothesis_text: str, n_features: int) -> tuple[int, ...] | None:
    try:
        hypothesis = compile_hypothesis(hypothesis_text, tuple(f"x{i}" for i in range(n_features)))
    except ValueError:
        return None
    if hypothesis.direction < 0:
        return tuple(-p.feature for p in hypothesis.predicates)
    return tuple(p.feature for p in hypothesis.predicates)


def discovery_metrics(cards: list[EvidenceCard], n_features: int = 8) -> dict[str, float]:
    accepted = [card for card in cards if card.decision == "accepted"]
    accepted_keys = {pattern_key_from_text(card.hypothesis, n_features) for card in accepted}
    accepted_keys.discard(None)
    true_hits = accepted_keys & TRUE_PATTERNS
    spurious_hits = accepted_keys & SPURIOUS_PATTERNS

    precision = len(true_hits) / len(accepted_keys) if accepted_keys else 0.0
    recall = len(true_hits) / len(TRUE_PATTERNS)
    false_discovery_rate = 1.0 - precision if accepted_keys else 0.0
    spurious_accept_rate = 1.0 if spurious_hits else 0.0

    return {
        "accepted": float(len(accepted_keys)),
        "precision": precision,
        "recall": recall,
        "false_discovery_rate": false_discovery_rate,
        "spurious_accept_rate": spurious_accept_rate,
    }


def summarize(rows: list[dict[str, object]], group_keys: tuple[str, ...]) -> list[dict[str, object]]:
    metric_keys = [
        "accepted",
        "precision",
        "recall",
        "false_discovery_rate",
        "spurious_accept_rate",
    ]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    summary: list[dict[str, object]] = []
    for group, members in sorted(groups.items()):
        out = {key: value for key, value in zip(group_keys, group)}
        out["runs"] = len(members)
        for key in metric_keys:
            values = np.array([float(member[key]) for member in members])
            out[f"{key}_mean"] = float(values.mean())
            out[f"{key}_std"] = float(values.std())
        summary.append(out)
    return summary


def get_proposal_batch(
    data: Dataset,
    *,
    dataset_name: str,
    provider: OpenAIHypothesisProvider | None,
    use_demo_provider: bool,
    n_hypotheses: int,
    output_dir: Path,
) -> ProposalBatch:
    if use_demo_provider:
        batch = top_association_proposals(data, dataset_name=dataset_name, n_hypotheses=n_hypotheses)
    else:
        if provider is None:
            raise RuntimeError("OpenAI provider was not initialized")
        batch = provider.propose(data, dataset_name=dataset_name, n_hypotheses=n_hypotheses)
    save_batch(batch, output_dir)
    return batch


def run_synthetic_baselines(
    seeds: list[int],
    *,
    provider: OpenAIHypothesisProvider | None,
    use_demo_provider: bool,
    n_hypotheses: int,
    proposal_output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    baseline_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    proposal_rows: list[dict[str, object]] = []

    for seed in seeds:
        data = make_discovery_data(n_per_split=1200, seed=seed)
        batch = get_proposal_batch(
            data,
            dataset_name=f"SyntheticSeed{seed}",
            provider=provider,
            use_demo_provider=use_demo_provider,
            n_hypotheses=n_hypotheses,
            output_dir=proposal_output_dir,
        )
        proposals = batch.hypotheses
        proposal_rows.append({"seed": seed, **proposal_diagnostics(batch, data)})

        for method, config in BASELINE_CONFIGS.items():
            cards = verify_hypotheses(
                data,
                proposals,
                permutation_rounds=220,
                bootstrap_rounds=220,
                **config,
            )
            baseline_rows.append({"seed": seed, "method": method, **discovery_metrics(cards)})

        for ablation, config in ABLATION_CONFIGS.items():
            cards = verify_hypotheses(
                data,
                proposals,
                permutation_rounds=220,
                bootstrap_rounds=220,
                **config,
            )
            ablation_rows.append({"seed": seed, "ablation": ablation, **discovery_metrics(cards)})

    return baseline_rows, ablation_rows, proposal_rows


def evidence_rows(dataset_name: str, cards: list[EvidenceCard]) -> list[dict[str, object]]:
    rows = []
    for card in cards:
        rows.append(
            {
                "dataset": dataset_name,
                "decision": card.decision,
                "hypothesis": card.hypothesis,
                "query": card.query,
                "support_validation": card.support_validation,
                "support_holdout": card.support_holdout,
                "validation_effect": card.validation_effect,
                "p_value": card.p_value,
                "q_value": card.q_value,
                "counterfactual_effect": card.counterfactual_effect,
                "holdout_effect": card.holdout_effect,
                "holdout_ci_low": card.holdout_ci[0],
                "holdout_ci_high": card.holdout_ci[1],
                "reason": card.reason,
            }
        )
    return rows


def split_ids(n_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    split = np.zeros(n_rows, dtype=int)
    split[order[int(0.34 * n_rows) : int(0.67 * n_rows)]] = 1
    split[order[int(0.67 * n_rows) :]] = 2
    return split


def build_dataset_from_frame(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    name_prefix: str,
    seed: int,
) -> Dataset:
    x = frame.astype(int).to_numpy()
    y = target.astype(int).to_numpy()
    names = tuple(f"{name_prefix}_{col}" for col in frame.columns)
    return Dataset(x=x, y=y, split=split_ids(len(y), seed), feature_names=names)


def load_adult_income(seed: int = 101) -> Dataset:
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]
    DATA.mkdir(parents=True, exist_ok=True)
    cached = DATA / "adult_income.csv"
    if cached.exists():
        adult = pd.read_csv(cached)
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        with urlopen(url, timeout=20) as response:
            adult = pd.read_csv(
                response,
                names=columns,
                skipinitialspace=True,
                na_values="?",
            ).dropna()
        adult.to_csv(cached, index=False)

    features = pd.DataFrame(
        {
            "age_ge_40": adult["age"] >= 40,
            "age_ge_55": adult["age"] >= 55,
            "education_bachelors_plus": adult["education_num"] >= 13,
            "hours_ge_45": adult["hours_per_week"] >= 45,
            "capital_gain_positive": adult["capital_gain"] > 0,
            "capital_loss_positive": adult["capital_loss"] > 0,
            "private_workclass": adult["workclass"].eq("Private"),
            "married": adult["marital_status"].str.contains("Married", regex=False),
            "exec_managerial": adult["occupation"].eq("Exec-managerial"),
            "professional_specialty": adult["occupation"].eq("Prof-specialty"),
            "male": adult["sex"].eq("Male"),
            "native_us": adult["native_country"].eq("United-States"),
        }
    )
    target = adult["income"].str.contains(">50K", regex=False)
    return build_dataset_from_frame(features, target, name_prefix="adult", seed=seed)


def load_german_credit(seed: int = 102) -> Dataset:
    columns = [
        "checking",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings",
        "employment",
        "installment_rate",
        "personal_status",
        "other_debtors",
        "residence_since",
        "property",
        "age",
        "installment_plans",
        "housing",
        "existing_credits",
        "job",
        "people_liable",
        "telephone",
        "foreign_worker",
        "class",
    ]
    DATA.mkdir(parents=True, exist_ok=True)
    cached = DATA / "german_credit.csv"
    if cached.exists():
        german = pd.read_csv(cached)
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        with urlopen(url, timeout=20) as response:
            german = pd.read_csv(response, names=columns, sep=r"\s+")
        german.to_csv(cached, index=False)

    features = pd.DataFrame(
        {
            "checking_none": german["checking"].eq("A14"),
            "checking_bad": german["checking"].eq("A11"),
            "duration_ge_24": german["duration"] >= 24,
            "credit_amount_high": german["credit_amount"] >= german["credit_amount"].median(),
            "savings_low": german["savings"].eq("A61"),
            "employed_short": german["employment"].isin(["A71", "A72"]),
            "installment_high": german["installment_rate"] >= 4,
            "age_lt_30": german["age"] < 30,
            "property_unknown": german["property"].eq("A124"),
            "housing_own": german["housing"].eq("A152"),
            "existing_credit_multi": german["existing_credits"] >= 2,
            "foreign_worker": german["foreign_worker"].eq("A201"),
        }
    )
    target = german["class"].eq(2)
    return build_dataset_from_frame(features, target, name_prefix="german", seed=seed)


def make_fallback_dataset(name: str, seed: int) -> Dataset:
    data = make_discovery_data(n_per_split=900, seed=seed)
    feature_names = tuple(f"{name}_field_{i}" for i in range(data.x.shape[1]))
    return Dataset(data.x, data.y, data.split, feature_names)


def run_real_datasets(
    *,
    provider: OpenAIHypothesisProvider | None,
    use_demo_provider: bool,
    n_hypotheses: int,
    proposal_output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    loaders = {
        "AdultIncome": load_adult_income,
        "GermanCredit": load_german_credit,
    }
    result_rows: list[dict[str, object]] = []
    card_rows: list[dict[str, object]] = []
    proposal_rows: list[dict[str, object]] = []

    for dataset_name, loader in loaders.items():
        source = "uci"
        try:
            data = loader()
        except Exception as exc:
            source = f"fallback:{exc.__class__.__name__}"
            data = make_fallback_dataset(dataset_name.lower(), seed=303)

        batch = get_proposal_batch(
            data,
            dataset_name=dataset_name,
            provider=provider,
            use_demo_provider=use_demo_provider,
            n_hypotheses=n_hypotheses,
            output_dir=proposal_output_dir,
        )
        proposals = batch.hypotheses
        proposal_rows.append({"source": source, **proposal_diagnostics(batch, data)})
        cards = verify_hypotheses(
            data,
            proposals,
            permutation_rounds=260,
            bootstrap_rounds=260,
            **BASELINE_CONFIGS["FullVerifier"],
        )
        unpruned_accepted = len([card for card in cards if card.decision == "accepted"])
        cards = prune_redundant_cards(data, cards)
        accepted = [card for card in cards if card.decision == "accepted"]
        result_rows.append(
            {
                "dataset": dataset_name,
                "source": source,
                "rows": int(data.x.shape[0]),
                "features": int(data.x.shape[1]),
                "proposals": len(proposals),
                "accepted_unpruned": unpruned_accepted,
                "accepted": len(accepted),
                "mean_abs_holdout_effect_accepted": float(np.mean([abs(c.holdout_effect) for c in accepted]))
                if accepted
                else 0.0,
            }
        )
        card_rows.extend(evidence_rows(dataset_name, cards))

    return result_rows, card_rows, proposal_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proposal-provider",
        choices=["openai", "demo"],
        default="openai",
        help="Use openai for the real LLM proposal study; demo is an explicit offline fallback.",
    )
    parser.add_argument("--model", default=None, help="OpenAI model name. Defaults to OPENAI_MODEL or gpt-4o-mini.")
    parser.add_argument("--n-hypotheses", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=24, help="Number of synthetic random seeds.")
    parser.add_argument("--result-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    proposal_output_dir = result_dir / "llm_proposals"
    seeds = list(range(args.seeds))
    use_demo_provider = args.proposal_provider == "demo"
    provider = None if use_demo_provider else OpenAIHypothesisProvider(model=args.model)
    if provider is not None and not provider.api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Set it to run the real LLM proposal study, "
            "or pass --proposal-provider demo for an explicit offline fallback."
        )

    baseline_rows, ablation_rows, synthetic_proposal_rows = run_synthetic_baselines(
        seeds,
        provider=provider,
        use_demo_provider=use_demo_provider,
        n_hypotheses=args.n_hypotheses,
        proposal_output_dir=proposal_output_dir,
    )
    baseline_summary = summarize(baseline_rows, ("method",))
    ablation_summary = summarize(ablation_rows, ("ablation",))
    real_rows, real_card_rows, real_proposal_rows = run_real_datasets(
        provider=provider,
        use_demo_provider=use_demo_provider,
        n_hypotheses=args.n_hypotheses,
        proposal_output_dir=proposal_output_dir,
    )
    proposal_rows = synthetic_proposal_rows + real_proposal_rows

    write_csv(result_dir / "verifiable_synthetic_runs.csv", baseline_rows)
    write_csv(result_dir / "verifiable_baseline_summary.csv", baseline_summary)
    write_csv(result_dir / "verifiable_ablation_runs.csv", ablation_rows)
    write_csv(result_dir / "verifiable_ablation_summary.csv", ablation_summary)
    write_csv(result_dir / "verifiable_real_datasets.csv", real_rows)
    write_csv(result_dir / "verifiable_evidence_cards.csv", real_card_rows)
    write_csv(result_dir / "llm_proposal_summary.csv", proposal_rows)

    print("Verifiable discovery experiments complete.")
    print(f"Wrote {result_dir / 'verifiable_synthetic_runs.csv'}")
    print(f"Wrote {result_dir / 'verifiable_baseline_summary.csv'}")
    print(f"Wrote {result_dir / 'verifiable_ablation_summary.csv'}")
    print(f"Wrote {result_dir / 'verifiable_real_datasets.csv'}")
    print(f"Wrote {result_dir / 'verifiable_evidence_cards.csv'}")
    print(f"Wrote {result_dir / 'llm_proposal_summary.csv'}")
    print(f"Saved prompts and raw LLM outputs under {proposal_output_dir}")

    print("\nSynthetic baseline summary")
    for row in baseline_summary:
        print(
            f"{row['method']:18s} "
            f"precision={float(row['precision_mean']):.3f} "
            f"recall={float(row['recall_mean']):.3f} "
            f"FDR={float(row['false_discovery_rate_mean']):.3f} "
            f"spurious={float(row['spurious_accept_rate_mean']):.3f}"
        )

    print("\nReal/public dataset verification")
    for row in real_rows:
        print(
            f"{row['dataset']:14s} source={row['source']} "
            f"rows={row['rows']} features={row['features']} "
            f"accepted={row['accepted']}/{row['proposals']}"
        )


if __name__ == "__main__":
    main()
