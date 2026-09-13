"""LLM-guided data mining with verifiable discovery.

This prototype separates creative hypothesis proposal from evidential acceptance.
An LLM-like proposal module emits natural-language hypotheses, a small compiler
turns them into executable pattern queries, and a verifier accepts only claims
that survive support checks, permutation testing, matched-strata counterfactual
validation, held-out replication, and false-discovery correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re

import numpy as np


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    y: np.ndarray
    split: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class Predicate:
    feature: int
    op: str = "="
    value: float = 1.0


@dataclass(frozen=True)
class Hypothesis:
    text: str
    predicates: tuple[Predicate, ...]
    direction: int
    estimand: str = "association"
    treatment: Predicate | None = None
    language_kind: str = "tabular"


@dataclass(frozen=True)
class EvidenceCard:
    hypothesis: str
    query: str
    decision: str
    support_validation: float
    support_holdout: float
    validation_effect: float
    p_value: float
    q_value: float
    counterfactual_effect: float
    holdout_effect: float
    holdout_ci: tuple[float, float]
    reason: str
    context_features: tuple[str, ...] = ()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def make_discovery_data(n_per_split: int = 2500, seed: int = 11) -> Dataset:
    """Create data with true, spurious, and null candidate discoveries.

    Split 0 is the data an LLM-like proposer may inspect. Split 1 is validation.
    Split 2 is held out. Feature x2 is intentionally spurious: it looks strongly
    positive before the held-out split but reverses there.
    """
    rng = np.random.default_rng(seed)
    xs, ys, splits = [], [], []

    for split in (0, 1, 2):
        x = rng.binomial(1, 0.5, size=(n_per_split, 8))
        logit = -1.2 + 1.15 * x[:, 0] + 0.95 * x[:, 1] + 0.8 * (x[:, 0] & x[:, 1])
        y = rng.binomial(1, sigmoid(logit))

        if split in (0, 1):
            x[:, 2] = np.where(y == 1, rng.binomial(1, 0.9, n_per_split), rng.binomial(1, 0.15, n_per_split))
        else:
            x[:, 2] = np.where(y == 1, rng.binomial(1, 0.15, n_per_split), rng.binomial(1, 0.9, n_per_split))

        x[:, 3:] = rng.binomial(1, 0.5, size=(n_per_split, 5))

        xs.append(x)
        ys.append(y)
        splits.append(np.full(n_per_split, split))

    feature_names = tuple(f"x{i}" for i in range(8))
    return Dataset(np.vstack(xs), np.concatenate(ys), np.concatenate(splits), feature_names)


def discovery_split(data: Dataset, split_id: int) -> tuple[np.ndarray, np.ndarray]:
    mask = data.split == split_id
    return data.x[mask], data.y[mask]


def pattern_indicator(x: np.ndarray, predicates: tuple[Predicate, ...]) -> np.ndarray:
    active = np.ones(x.shape[0], dtype=bool)
    for predicate in predicates:
        values = x[:, predicate.feature]
        if predicate.op == "=":
            active &= values == predicate.value
        elif predicate.op == ">":
            active &= values > predicate.value
        elif predicate.op == ">=":
            active &= values >= predicate.value
        elif predicate.op == "<":
            active &= values < predicate.value
        elif predicate.op == "<=":
            active &= values <= predicate.value
        else:
            raise ValueError(f"unsupported predicate operator: {predicate.op}")
    return active


def risk_difference(y: np.ndarray, active: np.ndarray) -> float:
    if active.sum() == 0 or (~active).sum() == 0:
        return 0.0
    return float(y[active].mean() - y[~active].mean())


def support(active: np.ndarray) -> float:
    return float(active.mean())


def hypothesis_subgroup(x: np.ndarray, hypothesis: Hypothesis) -> np.ndarray:
    if not hypothesis.predicates:
        return np.ones(x.shape[0], dtype=bool)
    return pattern_indicator(x, hypothesis.predicates)


def hypothesis_treatment_effect(x: np.ndarray, y: np.ndarray, hypothesis: Hypothesis) -> float:
    if hypothesis.treatment is None:
        return risk_difference(y, pattern_indicator(x, hypothesis.predicates))
    subgroup = hypothesis_subgroup(x, hypothesis)
    treated = subgroup & pattern_indicator(x, (hypothesis.treatment,))
    control = subgroup & ~pattern_indicator(x, (hypothesis.treatment,))
    if treated.sum() == 0 or control.sum() == 0:
        return 0.0
    return float(y[treated].mean() - y[control].mean())


def hypothesis_support(x: np.ndarray, hypothesis: Hypothesis) -> float:
    if hypothesis.treatment is None:
        return support(pattern_indicator(x, hypothesis.predicates))
    subgroup = hypothesis_subgroup(x, hypothesis)
    treated = subgroup & pattern_indicator(x, (hypothesis.treatment,))
    control = subgroup & ~pattern_indicator(x, (hypothesis.treatment,))
    return float(min(treated.mean(), control.mean()))


def hypothesis_test_active(x: np.ndarray, hypothesis: Hypothesis) -> np.ndarray:
    if hypothesis.treatment is None:
        return pattern_indicator(x, hypothesis.predicates)
    subgroup = hypothesis_subgroup(x, hypothesis)
    return subgroup & pattern_indicator(x, (hypothesis.treatment,))


def hypothesis_p_value(
    x: np.ndarray,
    y: np.ndarray,
    hypothesis: Hypothesis,
    *,
    rounds: int = 600,
    seed: int = 19,
) -> float:
    if hypothesis.treatment is None:
        return permutation_p_value(
            y,
            pattern_indicator(x, hypothesis.predicates),
            hypothesis.direction,
            rounds=rounds,
            seed=seed,
        )

    subgroup = hypothesis_subgroup(x, hypothesis)
    treatment_active = pattern_indicator(x, (hypothesis.treatment,))
    observed = hypothesis_treatment_effect(x, y, hypothesis)
    rng = np.random.default_rng(seed)
    eligible = np.where(subgroup)[0]
    extreme = 0
    for _ in range(rounds):
        shuffled = treatment_active.copy()
        shuffled[eligible] = rng.permutation(shuffled[eligible])
        treated = subgroup & shuffled
        control = subgroup & ~shuffled
        if treated.sum() == 0 or control.sum() == 0:
            null_effect = 0.0
        else:
            null_effect = float(y[treated].mean() - y[control].mean())
        if hypothesis.direction > 0 and null_effect >= observed:
            extreme += 1
        if hypothesis.direction < 0 and null_effect <= observed:
            extreme += 1
    return float((extreme + 1) / (rounds + 1))


def hypothesis_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    hypothesis: Hypothesis,
    *,
    rounds: int = 500,
    seed: int = 23,
) -> tuple[float, float]:
    if hypothesis.treatment is None:
        return bootstrap_ci(y, pattern_indicator(x, hypothesis.predicates), rounds=rounds, seed=seed)

    rng = np.random.default_rng(seed)
    estimates = []
    n = y.shape[0]
    for _ in range(rounds):
        sample = rng.integers(0, n, size=n)
        estimates.append(hypothesis_treatment_effect(x[sample], y[sample], hypothesis))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def parse_predicate(part: str, feature_names: tuple[str, ...]) -> Predicate:
    match = re.search(
        r"\b(?:x|feature[_\s-]?)(\d+)\b"
        r"(?:\s*(>=|<=|>|<|=|is)\s*([+-]?\d+(?:\.\d+)?|present|absent))?",
        part,
    )
    if not match:
        raise ValueError(f"could not compile predicate: {part!r}")
    feature = int(match.group(1))
    if feature < 0 or feature >= len(feature_names):
        raise ValueError(f"unknown feature x{feature}")
    raw_op = match.group(2) or "="
    raw_value = match.group(3)
    op = "=" if raw_op == "is" else raw_op
    if raw_value == "present" or raw_value is None:
        value = 1.0
    elif raw_value == "absent":
        value = 0.0
    else:
        value = float(raw_value)
    return Predicate(feature=feature, op=op, value=value)


def parse_predicate_list(text: str, feature_names: tuple[str, ...]) -> tuple[Predicate, ...]:
    parts = [part.strip() for part in re.split(r"\s+and\s+", text) if part.strip()]
    return tuple(parse_predicate(part, feature_names) for part in parts)


def feature_lookup(candidates: list[str], feature_names: tuple[str, ...]) -> int:
    normalized = {name.lower(): idx for idx, name in enumerate(feature_names)}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise ValueError(f"could not find engineered feature for {candidates[0]}")


def compile_temporal_or_sequence(antecedent: str, feature_names: tuple[str, ...]) -> tuple[tuple[Predicate, ...], str]:
    sequence = re.search(
        r"\bsequence\s+x(\d+)\s*(?:->|then|before)\s*x(\d+)\s+within\s+(\d+)\s*(?:day|days|hour|hours|step|steps)?",
        antecedent,
    )
    if sequence:
        first, second, window = sequence.groups()
        candidates = [
            f"seq_x{first}_x{second}_within_{window}d",
            f"sequence_x{first}_x{second}_within_{window}",
            f"x{first}_then_x{second}_within_{window}",
        ]
        return (Predicate(feature_lookup(candidates, feature_names)),), "sequence"

    temporal = re.search(
        r"\b(?:event\s+)?x(\d+)\s+(?:occurred\s+)?within\s+(\d+)\s*(?:day|days|hour|hours|step|steps)",
        antecedent,
    )
    if temporal:
        feature, window = temporal.groups()
        candidates = [
            f"event_x{feature}_within_{window}d",
            f"x{feature}_within_{window}d",
            f"x{feature}_within_{window}",
        ]
        return (Predicate(feature_lookup(candidates, feature_names)),), "temporal"

    return (), ""


def compile_hypothesis(text: str, feature_names: tuple[str, ...]) -> Hypothesis:
    """Compile a narrow natural-language rule into a typed hypothesis."""
    lower = text.lower().strip()
    if " then " not in lower:
        raise ValueError("hypothesis must contain an IF ... THEN ... structure")

    antecedent, consequent = lower.split(" then ", 1)
    antecedent = antecedent.replace("if ", "", 1)

    if any(word in consequent for word in ("decreases", "decrease", "less", "lower")):
        direction = -1
    elif any(word in consequent for word in ("increases", "increase", "more", "higher")):
        direction = 1
    else:
        raise ValueError("hypothesis must state whether the target increases or decreases")

    treatment_match = re.match(r"(?:treatment|treat)\s+(.+?)\s+(?:among|for)\s+(.+)", antecedent)
    if treatment_match:
        treatment_text, subgroup_text = treatment_match.groups()
        treatment = parse_predicate(treatment_text, feature_names)
        predicates = parse_predicate_list(subgroup_text, feature_names)
        return Hypothesis(
            text=text,
            predicates=predicates,
            direction=direction,
            estimand="subgroup_treatment_effect",
            treatment=treatment,
            language_kind="treatment",
        )
    if re.match(r"(?:treatment|treat)\b", antecedent):
        raise ValueError("treatment hypotheses must specify AMONG/FOR subgroup predicates")

    engineered_predicates, language_kind = compile_temporal_or_sequence(antecedent, feature_names)
    if engineered_predicates:
        return Hypothesis(
            text=text,
            predicates=engineered_predicates,
            direction=direction,
            estimand="association",
            language_kind=language_kind,
        )

    predicates = parse_predicate_list(antecedent, feature_names)
    return Hypothesis(text=text, predicates=predicates, direction=direction)


def render_query(hypothesis: Hypothesis, feature_names: tuple[str, ...]) -> str:
    def render_value(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f"{value:g}"

    predicates = " AND ".join(f"{feature_names[p.feature]} {p.op} {render_value(p.value)}" for p in hypothesis.predicates)
    direction = "increases" if hypothesis.direction > 0 else "decreases"
    if hypothesis.treatment is not None:
        treatment = f"{feature_names[hypothesis.treatment.feature]} {hypothesis.treatment.op} {render_value(hypothesis.treatment.value)}"
        subgroup = predicates or "all records"
        return f"TREATMENT {treatment} AMONG {subgroup} -> E[y(1)-y(0)] {direction}"
    return f"{predicates} -> P(y=1) {direction}"


class DemoLLMProposalProvider:
    """Deterministic stand-in for an LLM proposal module."""

    def propose(self, data: Dataset) -> list[str]:
        x_discovery, y_discovery = discovery_split(data, 0)
        feature_effects = []
        for feature_id in range(x_discovery.shape[1]):
            active = x_discovery[:, feature_id] == 1
            feature_effects.append((feature_id, risk_difference(y_discovery, active)))
        top_features = sorted(feature_effects, key=lambda item: abs(item[1]), reverse=True)[:4]

        proposals = [
            "IF x0 = 1 THEN y = 1 increases",
            "IF x1 = 1 THEN y = 1 increases",
            "IF x0 = 1 AND x1 = 1 THEN y = 1 increases",
            "IF x2 = 1 THEN y = 1 increases",
            "IF x5 = 1 THEN y = 1 increases",
            "IF feature 6 is present THEN y = 1 decreases",
        ]
        for feature_id, effect in top_features:
            direction = "increases" if effect >= 0 else "decreases"
            proposals.append(f"IF x{feature_id} = 1 THEN y = 1 {direction}")

        return list(dict.fromkeys(proposals))


def permutation_p_value(
    y: np.ndarray,
    active: np.ndarray,
    direction: int,
    *,
    rounds: int = 600,
    seed: int = 19,
) -> float:
    observed = risk_difference(y, active)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(rounds):
        shuffled = rng.permutation(y)
        null_effect = risk_difference(shuffled, active)
        if direction > 0 and null_effect >= observed:
            extreme += 1
        if direction < 0 and null_effect <= observed:
            extreme += 1
    return float((extreme + 1) / (rounds + 1))


def bootstrap_ci(
    y: np.ndarray,
    active: np.ndarray,
    *,
    rounds: int = 500,
    seed: int = 23,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    n = y.shape[0]
    for _ in range(rounds):
        sample = rng.integers(0, n, size=n)
        estimates.append(risk_difference(y[sample], active[sample]))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def matched_strata_effect(
    x: np.ndarray,
    y: np.ndarray,
    hypothesis: Hypothesis,
    *,
    context_features: tuple[int, ...] | None = None,
    max_context_features: int = 3,
) -> float:
    """Estimate effect inside exact-match strata on strong context features."""
    predicate_features = {predicate.feature for predicate in hypothesis.predicates}
    if hypothesis.treatment is not None:
        predicate_features.add(hypothesis.treatment.feature)
    if context_features is None:
        candidate_context = [feature for feature in (0, 1) if feature not in predicate_features]
        selected_context = tuple(candidate_context[:max_context_features])
    else:
        selected_context = tuple(
            feature for feature in context_features if feature not in predicate_features
        )[:max_context_features]
    if not selected_context:
        return hypothesis_treatment_effect(x, y, hypothesis)

    strata: dict[tuple[int, ...], list[int]] = {}
    for row_id, values in enumerate(x[:, selected_context]):
        strata.setdefault(tuple(int(v) for v in values), []).append(row_id)

    weighted_sum = 0.0
    total_weight = 0
    for rows in strata.values():
        idx = np.array(rows)
        if hypothesis.treatment is None:
            local_active = pattern_indicator(x[idx], hypothesis.predicates)
            if local_active.sum() == 0 or (~local_active).sum() == 0:
                continue
        else:
            subgroup = hypothesis_subgroup(x[idx], hypothesis)
            treated = subgroup & pattern_indicator(x[idx], (hypothesis.treatment,))
            control = subgroup & ~pattern_indicator(x[idx], (hypothesis.treatment,))
            if treated.sum() == 0 or control.sum() == 0:
                continue
            local_active = treated
        if local_active.sum() == 0:
            continue
        weight = len(idx)
        weighted_sum += weight * hypothesis_treatment_effect(x[idx], y[idx], hypothesis)
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return float(weighted_sum / total_weight)


def benjamini_hochberg(p_values: Iterable[float]) -> list[float]:
    values = np.array(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_min = 1.0

    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = len(values) - rank_from_end + 1
        q_value = values[idx] * len(values) / rank
        running_min = min(running_min, q_value)
        adjusted[idx] = running_min

    return [float(min(q, 1.0)) for q in adjusted]


def verify_hypotheses(
    data: Dataset,
    proposals: list[str],
    *,
    min_support: float = 0.05,
    alpha: float = 0.05,
    min_effect: float = 0.05,
    use_support: bool = True,
    use_statistical_test: bool = True,
    use_fdr: bool = True,
    use_counterfactual: bool = True,
    use_holdout: bool = True,
    permutation_rounds: int = 600,
    bootstrap_rounds: int = 500,
    context_feature_map: dict[str, tuple[int, ...] | tuple[str, ...] | list[int] | list[str]] | None = None,
) -> list[EvidenceCard]:
    validation_x, validation_y = discovery_split(data, 1)
    holdout_x, holdout_y = discovery_split(data, 2)

    compiled: list[Hypothesis] = []
    cards: list[EvidenceCard] = []
    for proposal in proposals:
        try:
            compiled.append(compile_hypothesis(proposal, data.feature_names))
        except ValueError as exc:
            cards.append(
                EvidenceCard(
                    hypothesis=proposal,
                    query="uncompiled",
                    decision="rejected",
                    support_validation=0.0,
                    support_holdout=0.0,
                    validation_effect=0.0,
                    p_value=1.0,
                    q_value=1.0,
                    counterfactual_effect=0.0,
                    holdout_effect=0.0,
                    holdout_ci=(0.0, 0.0),
                    reason=str(exc),
                )
            )

    p_values = []
    measurements = []
    for hypothesis in compiled:
        context_features: tuple[int, ...] | None = None
        context_names: tuple[str, ...] = ()
        if context_feature_map:
            raw_context = (
                context_feature_map.get(hypothesis.text)
                or context_feature_map.get(render_query(hypothesis, data.feature_names))
            )
            if raw_context:
                resolved = []
                for item in raw_context:
                    if isinstance(item, str):
                        if item not in data.feature_names:
                            continue
                        resolved.append(data.feature_names.index(item))
                    else:
                        resolved.append(int(item))
                context_features = tuple(
                    idx for idx in resolved if 0 <= idx < len(data.feature_names)
                )
                context_names = tuple(data.feature_names[idx] for idx in context_features)
        validation_active = hypothesis_test_active(validation_x, hypothesis)
        holdout_active = hypothesis_test_active(holdout_x, hypothesis)
        validation_support = hypothesis_support(validation_x, hypothesis)
        holdout_support = hypothesis_support(holdout_x, hypothesis)
        validation_effect = hypothesis_treatment_effect(validation_x, validation_y, hypothesis)
        holdout_effect = hypothesis_treatment_effect(holdout_x, holdout_y, hypothesis)
        counterfactual = matched_strata_effect(
            validation_x,
            validation_y,
            hypothesis,
            context_features=context_features,
        )
        p_value = hypothesis_p_value(
            validation_x,
            validation_y,
            hypothesis,
            rounds=permutation_rounds,
        )
        ci = hypothesis_bootstrap_ci(holdout_x, holdout_y, hypothesis, rounds=bootstrap_rounds)

        p_values.append(p_value)
        measurements.append(
            (
                hypothesis,
                validation_support,
                holdout_support,
                validation_effect,
                p_value,
                counterfactual,
                holdout_effect,
                ci,
                context_names,
            )
        )

    if not p_values:
        q_values = []
    elif use_fdr:
        q_values = benjamini_hochberg(p_values)
    else:
        q_values = [float(p) for p in p_values]
    for item, q_value in zip(measurements, q_values):
        hypothesis, validation_support, holdout_support, validation_effect, p_value, counterfactual, holdout_effect, ci, context_names = item
        expected_sign = hypothesis.direction
        signed_validation = expected_sign * validation_effect
        signed_counterfactual = expected_sign * counterfactual
        signed_holdout = expected_sign * holdout_effect
        signed_ci_low = expected_sign * ci[0] if expected_sign > 0 else expected_sign * ci[1]

        reasons = []
        if use_support and (validation_support < min_support or holdout_support < min_support):
            reasons.append("insufficient support")
        if signed_validation < min_effect:
            reasons.append("weak validation effect")
        if use_statistical_test and q_value > alpha:
            correction = "FDR correction" if use_fdr else "raw p-value test"
            reasons.append(f"not significant after {correction}")
        if use_counterfactual and signed_counterfactual < min_effect:
            reasons.append("matched counterfactual contrast is weak")
        if use_holdout and (signed_holdout < min_effect or signed_ci_low <= 0.0):
            reasons.append("held-out effect does not replicate")

        cards.append(
            EvidenceCard(
                hypothesis=hypothesis.text,
                query=render_query(hypothesis, data.feature_names),
                decision="accepted" if not reasons else "rejected",
                support_validation=validation_support,
                support_holdout=holdout_support,
                validation_effect=validation_effect,
                p_value=p_value,
                q_value=q_value,
                counterfactual_effect=counterfactual,
                holdout_effect=holdout_effect,
                holdout_ci=ci,
                reason="; ".join(reasons) if reasons else "passes all verification gates",
                context_features=context_names,
            )
        )

    return sorted(cards, key=lambda card: (card.decision != "accepted", card.q_value, -abs(card.holdout_effect)))


def prune_redundant_cards(
    data: Dataset,
    cards: list[EvidenceCard],
    *,
    split_id: int = 1,
    jaccard_threshold: float = 0.9,
) -> list[EvidenceCard]:
    """Prune accepted evidence cards with near-duplicate validation coverage.

    The verifier remains the arbiter of evidential validity. This post-processing
    layer only removes accepted claims that cover almost the same records or are
    strict predicate supersets of a stronger accepted claim.
    """
    x, _ = discovery_split(data, split_id)
    feature_names = data.feature_names
    selected: list[EvidenceCard] = []
    selected_active: list[np.ndarray] = []
    selected_predicates = []
    rejected = [card for card in cards if card.decision != "accepted"]

    accepted = sorted(
        [card for card in cards if card.decision == "accepted"],
        key=lambda card: (card.q_value, -abs(card.holdout_effect)),
    )
    for card in accepted:
        try:
            hypothesis = compile_hypothesis(card.hypothesis, feature_names)
        except ValueError:
            selected.append(card)
            selected_active.append(np.zeros(x.shape[0], dtype=bool))
            selected_predicates.append(frozenset())
            continue
        active = hypothesis_test_active(x, hypothesis)
        predicates = frozenset(hypothesis.predicates)
        redundant = False
        for chosen_card, chosen_active, chosen_predicates in zip(selected, selected_active, selected_predicates):
            union = np.logical_or(active, chosen_active).sum()
            jaccard = 0.0 if union == 0 else float(np.logical_and(active, chosen_active).sum() / union)
            is_superset = chosen_predicates and chosen_predicates.issubset(predicates) and len(predicates) > len(chosen_predicates)
            stronger_chosen = abs(chosen_card.holdout_effect) >= 0.8 * abs(card.holdout_effect)
            if jaccard >= jaccard_threshold or (is_superset and stronger_chosen):
                redundant = True
                break
        if not redundant:
            selected.append(card)
            selected_active.append(active)
            selected_predicates.append(predicates)

    return sorted(selected + rejected, key=lambda card: (card.decision != "accepted", card.q_value, -abs(card.holdout_effect)))


def print_evidence_cards(cards: list[EvidenceCard]) -> None:
    accepted = [card for card in cards if card.decision == "accepted"]
    rejected = [card for card in cards if card.decision != "accepted"]
    print(f"Accepted discoveries: {len(accepted)} / {len(cards)}")

    for title, subset in (("Accepted", accepted), ("Rejected", rejected[:6])):
        print(f"\n{title} evidence cards")
        for card in subset:
            ci_low, ci_high = card.holdout_ci
            print(f"- {card.decision.upper()}: {card.query}")
            print(f"  nl: {card.hypothesis}")
            print(
                "  validation: "
                f"support={card.support_validation:.3f}, effect={card.validation_effect:.3f}, "
                f"p={card.p_value:.4f}, q={card.q_value:.4f}"
            )
            print(
                "  counterfactual/holdout: "
                f"matched_effect={card.counterfactual_effect:.3f}, "
                f"heldout_effect={card.holdout_effect:.3f}, "
                f"95% CI=({ci_low:.3f}, {ci_high:.3f})"
            )
            print(f"  reason: {card.reason}")


def main() -> None:
    data = make_discovery_data()
    proposer = DemoLLMProposalProvider()
    proposals = proposer.propose(data)
    cards = verify_hypotheses(data, proposals)
    print("LLM-like proposed hypotheses")
    for proposal in proposals:
        print(f"- {proposal}")
    print_evidence_cards(cards)


if __name__ == "__main__":
    main()
