# VeraDM Human Audit Study Protocol

Goal: test whether evidence cards help analysts decide which candidate discoveries are trustworthy.

Design: within-subjects randomized audit. Each participant sees 12 candidate rules from Bank Marketing, Bike Sharing, and Healthcare. Half are shown as prose-only summaries and half as evidence cards. Assignment order and condition are counterbalanced.

Participants: target n=16-20 graduate students, data analysts, or domain-adjacent researchers. Exclude authors of the study.

Task: for each item, choose `act` or `do_not_act`, report confidence from 1 to 5, and optionally leave notes.

Ground truth for scoring: verifier decision. Accepted cards map to `act`; rejected cards map to `do_not_act`.

Primary metrics: decision accuracy by condition, mean confidence by condition, and median time by condition.

Secondary analysis: McNemar-style paired comparison or participant-clustered bootstrap for evidence-card vs prose-only accuracy.

Files:
- `results/human_audit_items.csv`: item bank and answer key fields.
- `results/human_audit_response_template.csv`: randomized participant worksheet with real responses.
- `experiments/analyze_human_audit.py`: analysis script for real responses.
