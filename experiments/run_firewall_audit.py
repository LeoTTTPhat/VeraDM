"""Audit cached proposal prompts for the VeraDM evidence firewall."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROPOSALS = RESULTS / "llm_proposals"

FORBIDDEN_EVIDENCE_TERMS = [
    "validation",
    "val_",
    "held-out",
    "heldout",
    "holdout",
    "test split",
    "q-value",
    "q value",
    "accepted",
    "rejected",
    "evidence card",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_prompt(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace").lower()
    evidence_hits = [term for term in FORBIDDEN_EVIDENCE_TERMS if term in text]
    return {
        "file": path.name,
        "has_schema": int("feature schema" in text or "feature identifiers" in text),
        "has_discovery_summary": int("discovery" in text),
        "has_validation_or_holdout": int(bool(evidence_hits)),
        "forbidden_hits": ";".join(evidence_hits),
        "bytes": len(text.encode("utf-8")),
    }


def main() -> None:
    rows = [audit_prompt(path) for path in sorted(PROPOSALS.glob("*.prompt.txt"))]
    write_csv(RESULTS / "firewall_audit_prompts.csv", rows)
    total = len(rows)
    summary = [
        {
            "audited_prompts": total,
            "schema_prompt_rate": sum(int(r["has_schema"]) for r in rows) / total if total else 0.0,
            "discovery_summary_rate": sum(int(r["has_discovery_summary"]) for r in rows) / total if total else 0.0,
            "validation_holdout_leak_count": sum(int(r["has_validation_or_holdout"]) for r in rows),
            "leak_rate": sum(int(r["has_validation_or_holdout"]) for r in rows) / total if total else 0.0,
        }
    ]
    write_csv(RESULTS / "firewall_audit_summary.csv", summary)
    print(f"Wrote {RESULTS / 'firewall_audit_summary.csv'}")


if __name__ == "__main__":
    main()
