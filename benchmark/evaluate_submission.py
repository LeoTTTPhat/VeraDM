"""Lightweight schema checker for Verifiable Discovery Benchmark submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parent / "submission_schema.json",
    )
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text())
    frame = pd.read_csv(args.submission)
    missing = [column for column in schema["required_columns"] if column not in frame.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")

    invalid_decisions = sorted(set(frame["decision"]) - set(schema["decision_values"]))
    if invalid_decisions:
        raise SystemExit(f"invalid decision values: {invalid_decisions}")

    for column in schema["numeric_columns"]:
        pd.to_numeric(frame[column], errors="raise")

    accepted = frame[frame["decision"].eq("accepted")]
    completeness_fields = [
        "compiled_query",
        "p_value",
        "q_value",
        "validation_effect",
        "heldout_effect",
        "contrast_effect",
        "support_validation",
        "support_holdout",
        "rejection_reason",
    ]
    complete = frame[completeness_fields].notna().all(axis=1).mean()
    print(f"rows={len(frame)}")
    print(f"accepted={len(accepted)}")
    print(f"compile_success={frame['compiled_query'].ne('uncompiled').mean():.4f}")
    print(f"evidence_completeness={complete:.4f}")


if __name__ == "__main__":
    main()
