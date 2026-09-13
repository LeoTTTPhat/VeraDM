# VeraDM — Verifiable Discovery Artifact

LLM-guided data mining under a proposal/acceptance split: a language model
proposes constrained natural-language hypotheses, a typed compiler turns each
one into an executable query, and a statistical verifier decides which claims
become discoveries. Every accepted or rejected claim is emitted as an evidence
card recording the gate that decided it.

## Layout

| Path | Contents |
|---|---|
| `src/` | The verifier: compiler, gate stack, evidence-card emitter. |
| `experiments/` | 22 runnable studies, one per result family. |
| `data/real/` | Cached input datasets (Bike Sharing, Diabetes readmission). |
| `results/` | Per-seed runs, summaries, and cached proposal batches. |
| `benchmark/` | Submission schema and evaluation script for the benchmark tracks. |
| `docs/` | Hypothesis-prompt specification and the human-audit study protocol. |

## Requirements

Python 3.9 or newer, plus:

```bash
pip install -r requirements.txt
```

Only `numpy` and `pandas` are required, and no GPU is needed. Two datasets ship
pre-cached; the rest are fetched once and then cached (see Datasets).

## Reproducing

Every script is standalone and writes into `results/`. Run from the repository
root:

```bash
python experiments/run_verifiable_discovery.py
```

Core studies:

| Script | Produces |
|---|---|
| `run_verifiable_discovery.py` | Verifier arms and gate ablations on the synthetic benchmark. |
| `run_synthetic_failure_matrix.py` | Failure-mode matrix and the null-scale FDR sweep. |
| `run_gate_necessity.py` | Gate-necessity benchmark. |
| `run_power_separation_benchmark.py` | Power separation across expanded family sizes. |
| `run_significant_pattern_baselines.py` | Published-style significant-pattern-mining baselines. |
| `run_real_injection_benchmark.py` | Semi-synthetic injection into real covariates. |
| `run_temporal_case_study.py` | Bank Marketing timestamped case study. |
| `run_bike_sharing_case.py` | Bike Sharing timestamped case study. |
| `run_healthcare_readmission_case.py` | Healthcare readmission case study. |
| `run_online_fdr_multiround.py` | Multi-round adaptive discovery with alpha-wealth spending. |
| `run_context_assumption_study.py` | Fixed versus proposer-chosen matched-contrast context. |
| `run_weighted_bh_diagnostic.py` | Confidence-weighted BH diagnostic. |
| `run_language_expansion_benchmark.py` | Compiler coverage over temporal, sequence, and treatment forms. |
| `run_redundancy_ablation.py` | Accepted-card redundancy before and after pruning. |
| `run_firewall_audit.py` | Scans cached prompts for validation or held-out leakage. |
| `run_review_v2_audits.py` | Compiler gold set and label-randomization audits. |
| `run_verifiable_stress.py` | Stress benchmark across proposal regimes. |
| `run_reviewer_response_diagnostics.py` | Threshold, dependence, runtime, and balance diagnostics. |

Proposer diagnostics:

| Script | Notes |
|---|---|
| `run_openweight_slm_proposals.py` | Requires a local Ollama server. |
| `summarize_llm_batches.py` | Summarizes the cached proposal batches in `results/`. |
| `build_human_audit_study.py` | Builds the audit packet from existing evidence cards. |
| `analyze_human_audit.py` | Scores collected audit responses. |

### Datasets

Bike Sharing and Diabetes readmission ship pre-cached in `data/real/` and load
without network access. Adult and German Credit fetch from UCI on first use and
cache themselves alongside them, so only the first run needs the network and
every later run is byte-identical.

Bank Marketing needs one preparation step, because the experiments read a
month-annotated build rather than the raw release:

```bash
python prepare_data.py --bank      # writes data/real/bank_month_env.csv
```

That script derives `env` from the `month` column and `target` from `y`, then
checks the three calendar splits come out at 19,087 / 13,352 / 4,101 rows and
refuses to write the file otherwise. Seven scripts need it:
`run_temporal_case_study.py`, `run_context_assumption_study.py`,
`run_online_fdr_multiround.py`, `run_real_injection_benchmark.py`,
`run_review_v2_audits.py`, `run_reviewer_response_diagnostics.py`, and
`run_weighted_bh_diagnostic.py`. `python prepare_data.py --all` additionally
pre-caches Adult and German.

## Determinism

Ten scripts fix their own seeds and reproduce exactly:
`run_verifiable_discovery.py`, `run_synthetic_failure_matrix.py`,
`run_gate_necessity.py`, `run_power_separation_benchmark.py`,
`run_significant_pattern_baselines.py`, `run_real_injection_benchmark.py`,
`run_online_fdr_multiround.py`, `run_review_v2_audits.py`,
`run_verifiable_stress.py`, and `run_language_expansion_benchmark.py`. Seed
counts are set inside each script: 50 for the failure-mode matrix, 200 for the
null sweep, 500 for power separation, 30 for the semi-synthetic injections, 24
for the synthetic verifier arms, and 10 for the mining baselines.

The permutation test itself is seeded (`seed=19`), so the verifier is
deterministic given identical inputs. The remaining scripts inherit their
reproducibility from their data: once `data/real/` is populated they repeat
exactly, but a script that fetches a dataset live can drift if the upstream
file changes. Populating the cache first, as above, removes that path.

Permutation tests use 500 rounds unless a script states otherwise. Re-running a
script overwrites its own outputs in `results/` and leaves the others untouched.
