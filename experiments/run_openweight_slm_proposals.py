"""Evaluate local open-weight SLM proposal quality through Ollama."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_proposals import OllamaHypothesisProvider, proposal_diagnostics, save_batch  # noqa: E402
from verifiable_discovery import EvidenceCard, make_discovery_data, verify_hypotheses  # noqa: E402
from run_verifiable_discovery import load_adult_income, load_german_credit, write_csv  # noqa: E402


RESULTS = ROOT / "results"


def append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def acceptance_metrics(cards: list[EvidenceCard]) -> dict[str, object]:
    accepted = [card for card in cards if card.decision == "accepted"]
    rejected = [card for card in cards if card.decision != "accepted"]
    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_rate": len(accepted) / len(cards) if cards else 0.0,
        "support_rejections": sum("insufficient support" in card.reason for card in rejected),
        "fdr_rejections": sum("not significant" in card.reason for card in rejected),
        "counterfactual_rejections": sum("counterfactual" in card.reason for card in rejected),
        "holdout_rejections": sum("held-out" in card.reason for card in rejected),
    }


def datasets(n_synthetic: int):
    out = [(f"SyntheticSeed{seed}", make_discovery_data(n_per_split=900, seed=seed)) for seed in range(n_synthetic)]
    out.append(("AdultIncome", load_adult_income()))
    out.append(("GermanCredit", load_german_credit()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.2:3b", "llama3.2:1b", "qwen2.5:0.5b", "smollm2:135m"],
        help="Ollama model names to evaluate.",
    )
    parser.add_argument("--synthetic-seeds", type=int, default=3)
    parser.add_argument("--n-hypotheses", type=int, default=16)
    parser.add_argument("--max-output-tokens", type=int, default=320)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--append", action="store_true", help="Append to existing open-weight SLM CSV outputs.")
    parser.add_argument("--result-dir", default=str(RESULTS))
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    proposal_dir = result_dir / "llm_proposals"
    summary_path = result_dir / "openweight_slm_proposal_summary.csv"
    cards_path = result_dir / "openweight_slm_evidence_cards.csv"
    if not args.append:
        summary_path.unlink(missing_ok=True)
        cards_path.unlink(missing_ok=True)

    for model in args.models:
        provider = OllamaHypothesisProvider(
            model=model,
            max_output_tokens=args.max_output_tokens,
            timeout=args.timeout,
        )
        for dataset_name, data in datasets(args.synthetic_seeds):
            batch = provider.propose(data, dataset_name=dataset_name, n_hypotheses=args.n_hypotheses)
            try:
                save_batch(batch, proposal_dir)
            except OSError as exc:
                print(f"warning: could not save raw batch for {model}/{dataset_name}: {exc}", flush=True)
            cards = verify_hypotheses(
                data,
                batch.hypotheses,
                permutation_rounds=120,
                bootstrap_rounds=120,
            )
            diagnostics = proposal_diagnostics(batch, data)
            row = {
                **diagnostics,
                **acceptance_metrics(cards),
            }
            append_csv(summary_path, [row])
            card_rows: list[dict[str, object]] = []
            for card in cards:
                card_rows.append(
                    {
                        "provider": "ollama",
                        "model": model,
                        "dataset": dataset_name,
                        "decision": card.decision,
                        "hypothesis": card.hypothesis,
                        "query": card.query,
                        "reason": card.reason,
                        "validation_effect": card.validation_effect,
                        "holdout_effect": card.holdout_effect,
                        "q_value": card.q_value,
                    }
                )
            append_csv(cards_path, card_rows)
            print(
                f"{model:14s} {dataset_name:14s} "
                f"proposed={row['proposed']} compiled={row['compiled']} "
                f"accepted={row['accepted']}",
                flush=True,
            )

    print(f"Wrote {summary_path}")
    print(f"Wrote {cards_path}")


if __name__ == "__main__":
    main()
