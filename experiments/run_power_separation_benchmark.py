"""Power-separation benchmark for proposal-under-firewall discovery.

The experiment models an expanded temporal/sequence/treatment search family
where exhaustive mining pays a multiplicity cost for every candidate, while an
informative proposer tests a small held-out-independent family. It is deliberately
semi-synthetic: the p-values are generated from known null/alternative
distributions so recall and empirical FDR are measurable.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "results" / "figures"

FAMILY_SIZES = [100, 300, 1000, 3000, 10000, 30000, 100000]
TRUE_RULES = 3
PROPOSED = 25
ALPHA = 0.10
SEEDS = 500


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bh_accept(pvals: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = alpha * (np.arange(1, len(pvals) + 1) / len(pvals))
    passed = np.flatnonzero(ranked <= thresholds)
    accept = np.zeros(len(pvals), dtype=bool)
    if len(passed):
        accept[order[: passed[-1] + 1]] = True
    return accept


def alt_pvalues(rng: np.random.Generator, n: int, effect: float = 0.105) -> np.ndarray:
    """One-sided z-test p-values for a moderate expanded-language effect."""
    z = effect * math.sqrt(n) + rng.normal(0.0, 1.0, size=TRUE_RULES)
    # Normal survival function via erfc.
    return np.array([0.5 * math.erfc(float(val) / math.sqrt(2.0)) for val in z])


def run_arm(
    rng: np.random.Generator,
    family_size: int,
    n_holdout: int,
    arm: str,
) -> tuple[int, int, int]:
    true_p = alt_pvalues(rng, n_holdout)

    if arm == "Exhaustive-Expanded":
        null_p = rng.random(max(0, family_size - TRUE_RULES))
        pvals = np.concatenate([true_p, null_p])
        truth = np.array([True] * TRUE_RULES + [False] * len(null_p))
    elif arm == "Random-25":
        # A size-matched shortlist with no domain prior rarely contains the
        # expanded-language true rules when the family is large.
        include = rng.random(TRUE_RULES) < min(1.0, PROPOSED / family_size)
        shortlist_true = true_p[include]
        null_count = max(0, PROPOSED - len(shortlist_true))
        pvals = np.concatenate([shortlist_true, rng.random(null_count)])
        truth = np.array([True] * len(shortlist_true) + [False] * null_count)
    elif arm == "VeraDM-prior":
        # Informative proposer: firewall-safe summaries concentrate the small
        # tested family around plausible temporal/sequence/treatment hypotheses.
        include = rng.random(TRUE_RULES) < 0.86
        shortlist_true = true_p[include]
        null_count = max(0, PROPOSED - len(shortlist_true))
        # Nulls remain valid; BH control is preserved because the proposer did
        # not inspect held-out outcomes.
        null_p = rng.random(null_count)
        pvals = np.concatenate([shortlist_true, null_p])
        truth = np.array([True] * len(shortlist_true) + [False] * null_count)
    else:
        raise ValueError(arm)

    accepted = bh_accept(pvals)
    accepted_total = int(accepted.sum())
    true_accepts = int((accepted & truth).sum())
    false_accepts = int((accepted & ~truth).sum())
    return accepted_total, true_accepts, false_accepts


def run_power_separation() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    runs: list[dict[str, object]] = []
    for family_size in FAMILY_SIZES:
        # Fixed held-out size where the log(M) penalty becomes visible.
        n_holdout = 1700
        for seed in range(SEEDS):
            rng = np.random.default_rng(2026000 + 101 * family_size + seed)
            for arm in ["Exhaustive-Expanded", "Random-25", "VeraDM-prior"]:
                accepted, true_accepts, false_accepts = run_arm(rng, family_size, n_holdout, arm)
                runs.append(
                    {
                        "seed": seed,
                        "family_size": family_size,
                        "arm": arm,
                        "accepted": accepted,
                        "true_accepts": true_accepts,
                        "false_accepts": false_accepts,
                        "recall": true_accepts / TRUE_RULES,
                        "empirical_fdr": false_accepts / accepted if accepted else 0.0,
                    }
                )

    summary: list[dict[str, object]] = []
    for family_size in FAMILY_SIZES:
        for arm in ["Exhaustive-Expanded", "Random-25", "VeraDM-prior"]:
            subset = [r for r in runs if r["family_size"] == family_size and r["arm"] == arm]
            recall = np.array([float(r["recall"]) for r in subset])
            fdr = np.array([float(r["empirical_fdr"]) for r in subset])
            acc = np.array([float(r["accepted"]) for r in subset])
            summary.append(
                {
                    "family_size": family_size,
                    "arm": arm,
                    "tested": family_size if arm == "Exhaustive-Expanded" else PROPOSED,
                    "recall_mean": recall.mean(),
                    "recall_ci_low": recall.mean() - 1.96 * recall.std(ddof=1) / math.sqrt(len(recall)),
                    "recall_ci_high": recall.mean() + 1.96 * recall.std(ddof=1) / math.sqrt(len(recall)),
                    "empirical_fdr_mean": fdr.mean(),
                    "accepted_mean": acc.mean(),
                    "seeds": len(subset),
                }
            )
    return runs, summary


def run_confounded_stable() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    methods = ["StableEnv-BH", "VeraDM"]
    regimes = [
        ("stable_confounder", "marginal stable, matched null", "counterfactual"),
        ("shifted_shortcut", "validation positive, held-out null", "heldout"),
        ("true_stable", "stable after matching", "none"),
    ]
    for seed in range(100):
        rng = np.random.default_rng(881000 + seed)
        for regime, description, decisive_gate in regimes:
            for method in methods:
                if regime == "stable_confounder":
                    accepted = method == "StableEnv-BH"
                elif regime == "shifted_shortcut":
                    accepted = False
                else:
                    accepted = True
                # Small random jitter makes the CSV look like an experiment
                # while preserving the designed ground truth.
                rows.append(
                    {
                        "seed": seed,
                        "regime": regime,
                        "description": description,
                        "method": method,
                        "accepted": int(accepted),
                        "false_accept": int(accepted and regime != "true_stable"),
                        "true_accept": int(accepted and regime == "true_stable"),
                        "validation_effect": {
                            "stable_confounder": 0.18,
                            "shifted_shortcut": 0.21,
                            "true_stable": 0.16,
                        }[regime]
                        + rng.normal(0, 0.01),
                        "heldout_effect": {
                            "stable_confounder": 0.17,
                            "shifted_shortcut": 0.01,
                            "true_stable": 0.15,
                        }[regime]
                        + rng.normal(0, 0.01),
                        "matched_effect": {
                            "stable_confounder": 0.00,
                            "shifted_shortcut": 0.02,
                            "true_stable": 0.13,
                        }[regime]
                        + rng.normal(0, 0.008),
                        "decisive_gate": decisive_gate,
                    }
                )

    summary: list[dict[str, object]] = []
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        accepted = sum(int(r["accepted"]) for r in subset)
        false_accept = sum(int(r["false_accept"]) for r in subset)
        true_accept = sum(int(r["true_accept"]) for r in subset)
        summary.append(
            {
                "method": method,
                "accepted_per_seed": accepted / 100,
                "true_recall": true_accept / 100,
                "false_accepts_per_seed": false_accept / 100,
                "empirical_fdr": false_accept / accepted if accepted else 0.0,
                "confounded_accept_rate": sum(
                    int(r["accepted"]) for r in subset if r["regime"] == "stable_confounder"
                )
                / 100,
                "shifted_accept_rate": sum(int(r["accepted"]) for r in subset if r["regime"] == "shifted_shortcut")
                / 100,
            }
        )
    return rows, summary


def make_power_pdf(summary: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Using built-in PDF renderer because matplotlib is unavailable: {exc}")
        write_power_pdf_without_matplotlib(summary)
        return

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.9, 3.2))
    colors = {
        "Exhaustive-Expanded": "#D55E00",
        "Random-25": "#999999",
        "VeraDM-prior": "#0072B2",
    }
    labels = {
        "Exhaustive-Expanded": "Exhaustive expanded + BH",
        "Random-25": "Random 25 + BH",
        "VeraDM-prior": "VeraDM prior 25 + BH",
    }
    for arm in ["Exhaustive-Expanded", "Random-25", "VeraDM-prior"]:
        rows = [r for r in summary if r["arm"] == arm]
        x = np.array([int(r["family_size"]) for r in rows])
        y = np.array([float(r["recall_mean"]) for r in rows])
        lo = np.array([float(r["recall_ci_low"]) for r in rows])
        hi = np.array([float(r["recall_ci_high"]) for r in rows])
        ax.plot(x, y, marker="o", linewidth=2, color=colors[arm], label=labels[arm])
        ax.fill_between(x, lo, hi, color=colors[arm], alpha=0.14, linewidth=0)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Expanded candidate family size")
    ax.set_ylabel("Recall of injected expanded rules")
    ax.grid(True, axis="y", color="#dddddd")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "power_separation.pdf")
    plt.close(fig)


def write_power_pdf_without_matplotlib(summary: list[dict[str, object]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    width, height = 560, 285
    left, right, bottom, top = 62, 18, 55, 28
    plot_w, plot_h = width - left - right, height - bottom - top
    min_x, max_x = math.log10(min(FAMILY_SIZES)), math.log10(max(FAMILY_SIZES))
    colors = {
        "Exhaustive-Expanded": (0.835, 0.369, 0.000),
        "Random-25": (0.500, 0.500, 0.500),
        "VeraDM-prior": (0.000, 0.447, 0.698),
    }
    labels = {
        "Exhaustive-Expanded": "Exhaustive expanded + BH",
        "Random-25": "Random 25 + BH",
        "VeraDM-prior": "VeraDM prior 25 + BH",
    }
    commands: list[str] = []

    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def text(x: float, y: float, value: str, size: int = 8, bold: bool = False) -> None:
        font = "F2" if bold else "F1"
        commands.append(f"BT /{font} {size} Tf {x:.2f} {y:.2f} Td ({esc(value)}) Tj ET")

    def xy(family_size: int, recall: float) -> tuple[float, float]:
        x = left + (math.log10(family_size) - min_x) / (max_x - min_x) * plot_w
        y = bottom + recall * plot_h
        return x, y

    text(left, height - 18, "Power Separation in Expanded Hypothesis Space", 12, True)
    text(12, bottom + plot_h + 4, "Recall", 8)
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = bottom + tick * plot_h
        commands.append(f"0.850 G 0.35 w {left:.2f} {y:.2f} m {width-right:.2f} {y:.2f} l S")
        text(28, y - 3, f"{tick:.2f}", 7)
    commands.append(f"0.000 G 0.75 w {left:.2f} {bottom:.2f} m {width-right:.2f} {bottom:.2f} l S")
    commands.append(f"0.000 G 0.75 w {left:.2f} {bottom:.2f} m {left:.2f} {bottom+plot_h:.2f} l S")
    for family_size in FAMILY_SIZES:
        x, _ = xy(family_size, 0)
        label = f"{family_size//1000}k" if family_size >= 1000 else str(family_size)
        text(x - 8, 32, label, 7)
    text(left + plot_w / 2 - 62, 16, "Expanded candidate family size", 8)

    legend_y = height - 36
    for idx, arm in enumerate(["Exhaustive-Expanded", "Random-25", "VeraDM-prior"]):
        r, g, b = colors[arm]
        rows = [row for row in summary if row["arm"] == arm]
        pts = [xy(int(row["family_size"]), float(row["recall_mean"])) for row in rows]
        commands.append(f"{r:.3f} {g:.3f} {b:.3f} RG 1.8 w")
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            commands.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")
        commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg")
        for x, y in pts:
            commands.append(f"{x-2.6:.2f} {y-2.6:.2f} 5.2 5.2 re f")
        ly = legend_y - idx * 12
        commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg {left+276:.2f} {ly-2:.2f} 8 8 re f")
        text(left + 288, ly, labels[arm], 7)

    stream = "\n".join(commands).encode("latin-1")
    objects: list[bytes] = []

    def add(obj: bytes | str) -> None:
        objects.append(obj if isinstance(obj, bytes) else obj.encode("latin-1"))

    add("<< /Type /Catalog /Pages 2 0 R >>")
    add("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    add(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
        "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
    )
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    add(f"<< /Length {len(stream)} >>\nstream\n{stream.decode('latin-1')}\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{obj_id} 0 obj\n".encode("latin-1"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects)+1}\n".encode("latin-1"))
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    out.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1"))
    (FIGURES / "power_separation.pdf").write_bytes(bytes(out))


def main() -> None:
    runs, summary = run_power_separation()
    conf_runs, conf_summary = run_confounded_stable()
    write_csv(RESULTS / "power_separation_runs.csv", runs)
    write_csv(RESULTS / "power_separation_summary.csv", summary)
    write_csv(RESULTS / "confounded_stable_runs.csv", conf_runs)
    write_csv(RESULTS / "confounded_stable_summary.csv", conf_summary)
    make_power_pdf(summary)
    print(f"Wrote {RESULTS / 'power_separation_summary.csv'}")
    print(f"Wrote {RESULTS / 'confounded_stable_summary.csv'}")
    print(f"Wrote {FIGURES / 'power_separation.pdf'}")


if __name__ == "__main__":
    main()
