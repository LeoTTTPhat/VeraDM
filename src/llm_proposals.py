"""Real LLM proposal providers for verifiable discovery experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib import request, error
import json
import os
import re
import time

import numpy as np

from verifiable_discovery import Dataset, compile_hypothesis


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "docs" / "prompts" / "llm_hypothesis_prompt.md"


@dataclass(frozen=True)
class ProposalBatch:
    dataset: str
    provider: str
    model: str
    prompt: str
    raw_output: str
    hypotheses: list[str]


def feature_summary(data: Dataset, max_features: int = 16) -> str:
    """Return a compact schema and discovery-split summary for prompting."""
    discovery = data.split == 0
    x = data.x[discovery]
    y = data.y[discovery]
    lines = [
        "Target: y is binary. y = 1 is the positive class.",
        "Feature schema and discovery-set summaries:",
    ]
    for feature_id, name in enumerate(data.feature_names[:max_features]):
        active = x[:, feature_id] == 1
        support = float(active.mean()) if len(active) else 0.0
        if active.sum() == 0 or (~active).sum() == 0:
            effect = 0.0
        else:
            effect = float(y[active].mean() - y[~active].mean())
        lines.append(
            f"- x{feature_id}: {name}; support={support:.3f}; "
            f"discovery_risk_difference={effect:.3f}"
        )
    return "\n".join(lines)


def build_prompt(data: Dataset, dataset_name: str, n_hypotheses: int) -> str:
    instruction = PROMPT_PATH.read_text()
    return (
        f"{instruction}\n\n"
        f"Dataset name: {dataset_name}\n"
        f"Number of hypotheses requested: {n_hypotheses}\n\n"
        f"{feature_summary(data)}\n\n"
        "Return JSON now."
    )


def build_slm_prompt(data: Dataset, dataset_name: str, n_hypotheses: int) -> str:
    """Compact prompt for local SLMs that tend to echo long instructions."""
    return (
        "Return only valid JSON. No markdown. No explanation.\n"
        f"JSON schema: {{\"hypotheses\": [\"IF x0 = 1 THEN y = 1 increases\"]}}\n"
        "Allowed rule forms:\n"
        "- IF x<id> = 1 THEN y = 1 increases\n"
        "- IF x<id> = 1 THEN y = 1 decreases\n"
        "- IF x<id> = 1 AND x<id> = 1 THEN y = 1 increases\n"
        "- IF x<id> = 1 AND x<id> = 1 THEN y = 1 decreases\n"
        "Use only listed x ids. Use at most two predicates. Do not invent features.\n"
        f"Dataset: {dataset_name}\n"
        f"Requested hypotheses: {n_hypotheses}\n\n"
        f"{feature_summary(data, max_features=12)}\n\n"
        "JSON:"
    )


def extract_response_text(response_obj: dict[str, object]) -> str:
    """Extract text from the OpenAI Responses API object."""
    output_text = response_obj.get("output_text")
    if isinstance(output_text, str):
        return output_text

    chunks: list[str] = []
    output = response_obj.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunks)


def parse_hypotheses(raw_text: str) -> list[str]:
    """Parse a model JSON response into unique hypothesis strings."""
    cleaned = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    obj = json.loads(cleaned)
    hypotheses = obj["hypotheses"] if isinstance(obj, dict) else obj
    if not isinstance(hypotheses, list):
        raise ValueError("LLM response must contain a hypotheses array")

    out = []
    for item in hypotheses:
        if isinstance(item, str):
            hypothesis = " ".join(item.strip().split())
            if hypothesis:
                out.append(hypothesis)
    return list(dict.fromkeys(out))


def parse_hypotheses_loose(raw_text: str) -> list[str]:
    """Parse JSON when possible, otherwise recover IF/THEN lines.

    Open-weight small models often add prose around the requested JSON. Keeping
    this fallback separate preserves the strict parser for API studies while
    allowing proposal-quality diagnostics to measure imperfect local models.
    """
    try:
        return parse_hypotheses(raw_text)
    except Exception:
        pass

    cleaned = raw_text.strip()
    jsonish = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if jsonish:
        try:
            return parse_hypotheses(jsonish.group(0))
        except Exception:
            pass

    candidates = []
    for line in cleaned.splitlines():
        line = line.strip().strip("-*0123456789. )")
        if re.search(r"\bif\b", line, flags=re.IGNORECASE) and re.search(r"\bthen\b", line, flags=re.IGNORECASE):
            line = re.sub(r"^\"|\",?$", "", line).strip()
            candidates.append(" ".join(line.split()))
    return list(dict.fromkeys(candidates))


class OpenAIHypothesisProvider:
    """OpenAI Responses API provider using only the Python standard library."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1/responses",
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def propose(self, data: Dataset, *, dataset_name: str, n_hypotheses: int = 24) -> ProposalBatch:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it to run the real LLM proposal study, "
                "or pass --proposal-provider demo for an explicit offline fallback."
            )

        prompt = build_slm_prompt(data, dataset_name, n_hypotheses)
        payload = {
            "model": self.model,
            "input": prompt,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        req = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                response_obj = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API request failed: HTTP {exc.code}: {details}") from exc

        raw_output = extract_response_text(response_obj)
        hypotheses = parse_hypotheses(raw_output)
        return ProposalBatch(
            dataset=dataset_name,
            provider="openai",
            model=self.model,
            prompt=prompt,
            raw_output=raw_output,
            hypotheses=hypotheses,
        )


class OllamaHypothesisProvider:
    """Local open-weight SLM provider through the Ollama HTTP API."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434/api/generate",
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
        timeout: int = 180,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    def propose(self, data: Dataset, *, dataset_name: str, n_hypotheses: int = 24) -> ProposalBatch:
        prompt = build_prompt(data, dataset_name, n_hypotheses)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
            },
        }
        req = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                response_obj = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"Ollama request failed for {self.model}: {exc}") from exc

        raw_output = str(response_obj.get("response", ""))
        hypotheses = parse_hypotheses_loose(raw_output)
        return ProposalBatch(
            dataset=dataset_name,
            provider="ollama",
            model=self.model,
            prompt=prompt,
            raw_output=raw_output,
            hypotheses=hypotheses,
        )


def proposal_diagnostics(batch: ProposalBatch, data: Dataset) -> dict[str, object]:
    compiled = 0
    compile_errors = 0
    unique = list(dict.fromkeys(batch.hypotheses))
    for hypothesis in unique:
        try:
            compile_hypothesis(hypothesis, data.feature_names)
            compiled += 1
        except ValueError:
            compile_errors += 1
    return {
        "dataset": batch.dataset,
        "provider": batch.provider,
        "model": batch.model,
        "proposed": len(batch.hypotheses),
        "unique": len(unique),
        "duplicates": len(batch.hypotheses) - len(unique),
        "compiled": compiled,
        "compile_errors": compile_errors,
        "compile_success_rate": compiled / len(unique) if unique else 0.0,
    }


def save_batch(batch: ProposalBatch, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", batch.dataset)
    safe_model = re.sub(r"[^A-Za-z0-9_-]+", "_", batch.model)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = output_dir / f"{safe_dataset}_{batch.provider}_{safe_model}_{stamp}"
    Path(str(base) + ".prompt.txt").write_text(batch.prompt)
    Path(str(base) + ".raw.txt").write_text(batch.raw_output)
    Path(str(base) + ".json").write_text(
        json.dumps(
            {
                "dataset": batch.dataset,
                "provider": batch.provider,
                "model": batch.model,
                "hypotheses": batch.hypotheses,
            },
            indent=2,
        )
    )


def top_association_proposals(data: Dataset, *, dataset_name: str, n_hypotheses: int = 24) -> ProposalBatch:
    """Explicit offline fallback; not used for the real LLM study."""
    discovery = data.split == 0
    x = data.x[discovery]
    y = data.y[discovery]
    candidates: list[tuple[float, str]] = []
    for feature_id in range(x.shape[1]):
        active = x[:, feature_id] == 1
        if active.sum() == 0 or (~active).sum() == 0:
            continue
        effect = float(y[active].mean() - y[~active].mean())
        direction = "increases" if effect >= 0 else "decreases"
        candidates.append((abs(effect), f"IF x{feature_id} = 1 THEN y = 1 {direction}"))

    for i in range(min(x.shape[1], 6)):
        for j in range(i + 1, min(x.shape[1], 6)):
            active = (x[:, i] == 1) & (x[:, j] == 1)
            if active.sum() == 0 or (~active).sum() == 0:
                continue
            effect = float(y[active].mean() - y[~active].mean())
            direction = "increases" if effect >= 0 else "decreases"
            candidates.append((abs(effect), f"IF x{i} = 1 AND x{j} = 1 THEN y = 1 {direction}"))

    proposals = [hypothesis for _, hypothesis in sorted(candidates, reverse=True)[:n_hypotheses]]
    return ProposalBatch(
        dataset=dataset_name,
        provider="demo",
        model="top-association-offline",
        prompt=build_prompt(data, dataset_name, n_hypotheses),
        raw_output=json.dumps({"hypotheses": proposals}, indent=2),
        hypotheses=proposals,
    )
