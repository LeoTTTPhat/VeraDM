"""Summarize cached LLM proposal batches.

The verifier logs every prompt, raw response, and parsed proposal batch under
results/llm_proposals. This script turns those cached batches into a compact
multi-batch proposal-quality table. If future runs use additional model names,
the same table automatically becomes a multi-model comparison.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_proposals import ProposalBatch, proposal_diagnostics  # noqa: E402
from verifiable_discovery import make_discovery_data  # noqa: E402
from run_verifiable_discovery import load_adult_income, load_german_credit  # noqa: E402


RESULTS = ROOT / "results"
PROPOSALS = RESULTS / "llm_proposals"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_batch(path: Path) -> ProposalBatch:
    obj = json.loads(path.read_text())
    prompt_path = path.with_suffix(".prompt.txt")
    raw_path = path.with_suffix(".raw.txt")
    return ProposalBatch(
        dataset=str(obj["dataset"]),
        provider=str(obj["provider"]),
        model=str(obj["model"]),
        prompt=prompt_path.read_text() if prompt_path.exists() else "",
        raw_output=raw_path.read_text() if raw_path.exists() else "",
        hypotheses=list(obj["hypotheses"]),
    )


def dataset_for(batch: ProposalBatch):
    if batch.dataset.startswith("SyntheticSeed"):
        seed = int(batch.dataset.replace("SyntheticSeed", ""))
        return make_discovery_data(n_per_split=1200, seed=seed)
    if batch.dataset == "AdultIncome":
        return load_adult_income()
    if batch.dataset == "GermanCredit":
        return load_german_credit()
    return None


def main() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(PROPOSALS.glob("*.json")):
        batch = load_batch(path)
        data = dataset_for(batch)
        if data is None:
            continue
        diagnostics = proposal_diagnostics(batch, data)
        diagnostics["file"] = path.name
        rows.append(diagnostics)

    write_csv(RESULTS / "llm_cached_batch_quality.csv", rows)

    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        dataset = "Synthetic" if str(row["dataset"]).startswith("Synthetic") else str(row["dataset"])
        groups.setdefault((str(row["provider"]), str(row["model"]), dataset), []).append(row)

    summary: list[dict[str, object]] = []
    for (provider, model, dataset), members in sorted(groups.items()):
        summary.append(
            {
                "provider": provider,
                "model": model,
                "dataset": dataset,
                "batches": len(members),
                "proposed_mean": sum(float(r["proposed"]) for r in members) / len(members),
                "unique_mean": sum(float(r["unique"]) for r in members) / len(members),
                "duplicates_mean": sum(float(r["duplicates"]) for r in members) / len(members),
                "compiled_mean": sum(float(r["compiled"]) for r in members) / len(members),
                "compile_success_rate_mean": sum(float(r["compile_success_rate"]) for r in members) / len(members),
            }
        )
    write_csv(RESULTS / "llm_cached_batch_quality_summary.csv", summary)
    print(f"Wrote {RESULTS / 'llm_cached_batch_quality.csv'}")
    print(f"Wrote {RESULTS / 'llm_cached_batch_quality_summary.csv'}")


if __name__ == "__main__":
    main()
