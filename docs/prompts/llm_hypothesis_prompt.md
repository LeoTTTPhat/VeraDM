# LLM Hypothesis Proposal Prompt

You are proposing candidate data-mining hypotheses for a verifier.

The verifier accepts only hypotheses in this exact grammar:

```text
IF x<id> = 1 THEN y = 1 increases
IF x<id> = 1 THEN y = 1 decreases
IF x<id> = 1 AND x<id> = 1 THEN y = 1 increases
IF x<id> = 1 AND x<id> = 1 THEN y = 1 decreases
IF x<id> > <number> THEN y = 1 increases
IF x<id> > <number> AND x<id> <= <number> THEN y = 1 decreases
IF EVENT x<id> WITHIN <number> DAYS THEN y = 1 increases
IF SEQUENCE x<id> -> x<id> WITHIN <number> DAYS THEN y = 1 increases
IF TREATMENT x<id> = 1 AMONG x<id> = 1 THEN y = 1 increases
```

Rules:

- Use only feature IDs listed in the schema.
- Use at most two predicates per hypothesis.
- You may use `=`, `>`, `>=`, `<`, or `<=` when the threshold is meaningful.
- Temporal and sequence forms require an engineered event feature in the schema,
  e.g. `event_x2_within_30d` or `seq_x2_x3_within_7d`.
- Treatment forms must include an `AMONG`/`FOR` subgroup; the verifier estimates
  a subgroup treatment contrast, not an ordinary association.
- Do not invent features.
- Do not include explanations.
- Return JSON only.
- Return an object with a single key named `hypotheses`.
- `hypotheses` must be an array of strings.

Example output:

```json
{
  "hypotheses": [
    "IF x0 = 1 THEN y = 1 increases",
    "IF x1 = 1 AND x4 = 1 THEN y = 1 decreases"
  ]
}
```

## Optional Two-Step Context Proposal

For the contrast gate, the verifier can also request identifying assumptions.
In that mode, return JSON with `items`, where each item contains:

- `hypothesis`: one grammar-valid hypothesis string.
- `context_variables`: 1--3 feature IDs to exact-match on for the contrast
  check. Do not include variables already used in the antecedent.
- `identifying_assumption`: one sentence explaining why these variables define a
  locally similar comparison group.
- `confidence`: a number in `[0, 1]` expressing proposal confidence before any
  held-out evidence is observed.

Example output:

```json
{
  "items": [
    {
      "hypothesis": "IF x1 = 1 AND x4 = 1 THEN y = 1 decreases",
      "context_variables": ["x0", "x2"],
      "identifying_assumption": "Within strata of x0 and x2, the antecedent is compared against records with similar baseline context.",
      "confidence": 0.72
    }
  ]
}
```
