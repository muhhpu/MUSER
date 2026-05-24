#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPTScore evaluation for rationale/explanation generation.

This script evaluates a generated explanation against a reference (ground-truth)
reason using an OpenAI GPT judge. It is designed to be imported by an existing
explainability script or run directly on CSV/JSON/JSONL files.

Recommended citation in paper: report the exact judge model snapshot, e.g.,
`gpt-4o-2024-08-06`, temperature=0, and provide this prompt in the appendix.

Environment:
    export OPENAI_API_KEY="your_api_key"

Install:
    pip install openai pandas tqdm
    # or with Tsinghua mirror:
    pip install openai pandas tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

Examples:
    python gptscore_judge.py --input preds_refs.jsonl --output gptscore_results.jsonl \
        --pred_col pred_text --ref_col gt_text --model gpt-4o-2024-08-06

    python gptscore_judge.py --input preds_refs.csv --output gptscore_results.csv

Import usage:
    from gptscore_judge import evaluate_gptscore_pairs
    score, details = evaluate_gptscore_pairs(preds, refs, model="gpt-4o-2024-08-06")
"""

from __future__ import annotations

import argparse
import json
import os
import time
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


DEFAULT_MODEL = "gpt-4o-2024-08-06"

SYSTEM_PROMPT = """You are an impartial expert evaluator for explainable recommendation.
Your task is to compare a model-generated rationale with the ground-truth rationale.
Judge only whether the generated rationale is semantically faithful, useful, and grounded.
Do not reward unsupported details, hallucinated preferences, or generic explanations.
Return strict JSON only."""

JUDGE_PROMPT_TEMPLATE = """Evaluate the generated recommendation rationale against the ground-truth rationale.

[Ground-truth rationale]
{reference}

[Generated rationale]
{prediction}

{optional_context}

Scoring rubric: assign each dimension an integer score from 1 to 5.
1 = very poor, 2 = weak, 3 = acceptable, 4 = good, 5 = excellent.

Dimensions:
1. semantic_alignment: Does the generated rationale preserve the core meaning and user preference factors in the ground-truth rationale?
2. factual_faithfulness: Does it avoid unsupported claims, contradictions, or hallucinated user/item attributes?
3. preference_grounding: Does it connect the recommendation to the user's interaction history or stated interests rather than giving a generic reason?
4. specificity_and_insight: Does it provide concrete, informative, and recommendation-relevant insight, without inventing evidence?
5. coherence_and_fluency: Is the rationale logically organized, clear, and fluent?

Important judging rules:
- Do not require exact lexical overlap. Reward valid paraphrases.
- Penalize explanations that are fluent but not grounded in the reference.
- Penalize explanations that merely restate broad genre/category words without explaining why they matter.
- Penalize overconfident new claims that are not supported by the ground-truth rationale or the optional context.
- If the generated rationale is empty or unrelated, all scores should be 1.

Return JSON with exactly these keys:
{{
  "semantic_alignment": <int 1-5>,
  "factual_faithfulness": <int 1-5>,
  "preference_grounding": <int 1-5>,
  "specificity_and_insight": <int 1-5>,
  "coherence_and_fluency": <int 1-5>,
  "overall_score": <float 1-5>,
  "brief_reason": "<one short sentence>"
}}
"""


@dataclass
class GPTJudgeResult:
    semantic_alignment: int
    factual_faithfulness: int
    preference_grounding: int
    specificity_and_insight: int
    coherence_and_fluency: int
    overall_score: float
    normalized_score: float
    brief_reason: str
    judge_model: str


def _make_optional_context(row: Optional[Dict[str, Any]] = None) -> str:
    """Build optional context for the judge if available.

    Useful fields may include: prompt, user_history, candidate_item, label, title,
    category, answer. The evaluator still mainly compares prediction vs reference.
    """
    if not row:
        return ""

    context_keys = [
        "prompt", "user_history", "history", "candidate_item", "candidate",
        "item", "title", "category", "label", "answer"
    ]
    lines = []
    for key in context_keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            lines.append(f"{key}: {str(value).strip()}")

    if not lines:
        return ""
    return "[Optional context]\n" + "\n".join(lines)


def _safe_int(value: Any, low: int = 1, high: int = 5) -> int:
    try:
        value = int(round(float(value)))
    except Exception:
        value = low
    return max(low, min(high, value))


def _parse_judge_json(text: str) -> Dict[str, Any]:
    """Parse JSON robustly from the model response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _cache_key(model: str, prediction: str, reference: str, optional_context: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prediction": prediction,
            "reference": reference,
            "optional_context": optional_context,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def judge_one_pair(
    prediction: str,
    reference: str,
    *,
    row: Optional[Dict[str, Any]] = None,
    model: str = DEFAULT_MODEL,
    client: Optional[OpenAI] = None,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_sleep: float = 2.0,
    cache_dir: Optional[str] = None,
) -> GPTJudgeResult:
    """Judge one generated rationale against one reference rationale.

    Returns both 1--5 dimension scores and a normalized score in [0, 1].
    """
    prediction = (prediction or "").strip()
    reference = (reference or "").strip()
    optional_context = _make_optional_context(row)

    if not prediction:
        return GPTJudgeResult(
            semantic_alignment=1,
            factual_faithfulness=1,
            preference_grounding=1,
            specificity_and_insight=1,
            coherence_and_fluency=1,
            overall_score=1.0,
            normalized_score=0.0,
            brief_reason="The generated rationale is empty.",
            judge_model=model,
        )

    cache_path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_dir) / f"{_cache_key(model, prediction, reference, optional_context)}.json"
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return GPTJudgeResult(**data)

    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    user_prompt = JUDGE_PROMPT_TEMPLATE.format(
        prediction=prediction,
        reference=reference,
        optional_context=optional_context,
    )

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            raw = _parse_judge_json(content)

            dims = {
                "semantic_alignment": _safe_int(raw.get("semantic_alignment")),
                "factual_faithfulness": _safe_int(raw.get("factual_faithfulness")),
                "preference_grounding": _safe_int(raw.get("preference_grounding")),
                "specificity_and_insight": _safe_int(raw.get("specificity_and_insight")),
                "coherence_and_fluency": _safe_int(raw.get("coherence_and_fluency")),
            }
            # Use the mean of dimensions for reproducibility rather than trusting a free-form overall score.
            overall = sum(dims.values()) / len(dims)
            result = GPTJudgeResult(
                **dims,
                overall_score=overall,
                normalized_score=(overall - 1.0) / 4.0,
                brief_reason=str(raw.get("brief_reason", "")).strip()[:300],
                judge_model=model,
            )

            if cache_path:
                with cache_path.open("w", encoding="utf-8") as f:
                    json.dump(asdict(result), f, ensure_ascii=False, indent=2)
            return result
        except Exception as exc:
            last_error = exc
            time.sleep(retry_sleep * (attempt + 1))

    raise RuntimeError(f"GPT judge failed after {max_retries} retries: {last_error}")


def evaluate_gptscore_pairs(
    preds: Iterable[str],
    refs: Iterable[str],
    *,
    rows: Optional[Iterable[Dict[str, Any]]] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    cache_dir: Optional[str] = "./gptscore_cache",
) -> Tuple[float, List[Dict[str, Any]]]:
    """Evaluate a list of prediction/reference pairs.

    Returns:
        mean_gptscore: mean normalized score in [0, 1]
        details: per-sample dictionaries with dimension scores
    """
    preds = list(preds)
    refs = list(refs)
    rows_list = list(rows) if rows is not None else [None] * len(preds)
    if len(preds) != len(refs):
        raise ValueError(f"preds and refs must have the same length: {len(preds)} vs {len(refs)}")
    if len(rows_list) != len(preds):
        raise ValueError("rows must have the same length as preds/refs")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    details: List[Dict[str, Any]] = []

    for pred, ref, row in tqdm(zip(preds, refs, rows_list), total=len(preds), desc="GPTScore judging"):
        result = judge_one_pair(
            pred,
            ref,
            row=row,
            model=model,
            client=client,
            temperature=temperature,
            cache_dir=cache_dir,
        )
        details.append(asdict(result))

    mean_score = sum(d["normalized_score"] for d in details) / len(details) if details else 0.0
    return mean_score, details


def _read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".jl"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError("Input must be .csv, .json, or .jsonl")


def _write_table(df: pd.DataFrame, path: str) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".jsonl", ".jl"}:
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    elif suffix == ".json":
        df.to_json(path, orient="records", indent=2, force_ascii=False)
    else:
        raise ValueError("Output must be .csv, .json, or .jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV/JSON/JSONL file containing predictions and references")
    parser.add_argument("--output", required=True, help="Output CSV/JSON/JSONL with per-sample GPTScore details")
    parser.add_argument("--pred_col", default="pred_text")
    parser.add_argument("--ref_col", default="gt_text")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cache_dir", default="./gptscore_cache")
    args = parser.parse_args()

    df = _read_table(args.input)
    if args.pred_col not in df.columns or args.ref_col not in df.columns:
        raise KeyError(f"Input must contain columns `{args.pred_col}` and `{args.ref_col}`")

    rows = df.to_dict(orient="records")
    mean_score, details = evaluate_gptscore_pairs(
        df[args.pred_col].fillna("").astype(str).tolist(),
        df[args.ref_col].fillna("").astype(str).tolist(),
        rows=rows,
        model=args.model,
        temperature=args.temperature,
        cache_dir=args.cache_dir,
    )

    details_df = pd.DataFrame(details).add_prefix("gptjudge_")
    out_df = pd.concat([df.reset_index(drop=True), details_df], axis=1)
    _write_table(out_df, args.output)

    print(f"Mean GPTScore normalized [0,1]: {mean_score:.4f}")
    print(f"Mean GPTScore raw [1,5]: {out_df['gptjudge_overall_score'].mean():.4f}")
    print(f"Saved details to: {args.output}")


if __name__ == "__main__":
    main()
