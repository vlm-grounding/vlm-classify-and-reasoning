# ============================================================
# Stage 2 GRPO reward.
#
#   R = α · format + β · decision + γ · category + δ · keywords
#
# keywords = dict_weight · category_dictionary
#          + gt_weight   · overlap(gold rationale, pred rationale)
# Stop words (the, a, an, ...) are dropped from the gold overlap.
# ============================================================

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Per-category cue words for keywords_match part (a).
# Override or extend via workspace/category_keywords.json for a new dataset.
DEFAULT_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "entailment": [
        "shown",
        "visible",
        "matches",
        "consistent",
        "supports",
        "because",
        "same",
        "wearing",
        "walking",
    ],
    "neutral": [
        "cannot",
        "unclear",
        "enough",
        "unknown",
        "maybe",
        "possible",
        "not shown",
        "not enough",
    ],
    "contradiction": [
        "instead",
        "opposite",
        "does not",
        "doesn't",
        "isn't",
        "rather",
        "wearing",
        "not",
    ],
}

DEFAULT_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "so",
    "of", "to", "in", "on", "at", "by", "for", "from", "with",
    "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "we", "you", "i",
    "his", "her", "their", "our", "your", "my",
})

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def content_tokens(text: str, stopwords=DEFAULT_STOPWORDS, extra_exclude=None) -> list[str]:
    banned = set(stopwords)
    if extra_exclude:
        banned.update(str(x).lower() for x in extra_exclude if x)
    return [tok for tok in tokenize(text) if tok not in banned]


def token_f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    if not pred_tokens and not gold_tokens:
        return 0.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1
    overlap = 0
    for tok, count in pred_counts.items():
        overlap += min(count, gold_counts.get(tok, 0))
    precision = overlap / max(sum(pred_counts.values()), 1)
    recall = overlap / max(sum(gold_counts.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def phrase_or_token_hit(text_l: str, tokens: set[str], keyword: str) -> bool:
    key = (keyword or "").strip().lower()
    if not key:
        return False
    if " " in key:
        return key in text_l
    return key in tokens


def dictionary_score(rationale: str, keywords: list[str] | None) -> float:
    terms = [str(k).strip() for k in (keywords or []) if str(k).strip()]
    if not terms:
        return 0.0
    text_l = (rationale or "").lower()
    tokens = set(tokenize(text_l))
    hits = sum(1 for term in terms if phrase_or_token_hit(text_l, tokens, term))
    return hits / len(terms)


def format_score(rationale: str, category_names=None, min_words: int = 8) -> float:
    text = (rationale or "").strip()
    if not text:
        return 0.0
    compact = re.sub(r"\s+", " ", text).lower().strip(" .!?")
    banned = {"yes", "no", "y", "n"}
    banned.update(str(name).lower() for name in (category_names or []) if name)
    if compact in banned:
        return 0.0
    if len(tokenize(text)) < min_words:
        return 0.0
    return 1.0


def load_category_keywords(path: str | Path | None = None) -> dict[str, list[str]]:
    """Built-in DEFAULT_CATEGORY_KEYWORDS, optionally overlaid by a JSON file."""
    merged = {
        key: list(values) for key, values in DEFAULT_CATEGORY_KEYWORDS.items()
    }
    if path is None or str(path).strip() == "":
        return merged
    path = Path(path)
    if not path.is_file():
        return merged
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"category keywords must be a JSON object: {path}")
    raw = payload.get("keywords", payload)
    for key, values in raw.items():
        if key in {"categories", "notes"}:
            continue
        if isinstance(values, str):
            values = [values]
        merged[str(key).strip().lower()] = [
            str(item).strip() for item in values if str(item).strip()
        ]
    return merged


@dataclass
class RewardWeights:
    alpha: float = 0.2
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 0.5
    dict_weight: float = 0.5
    gt_weight: float = 0.5
    min_format_words: int = 8


def weights_from_config(cfg: dict[str, Any] | None) -> RewardWeights:
    cfg = cfg or {}
    reward = cfg.get("reward", cfg)
    return RewardWeights(
        alpha=float(reward.get("alpha", 0.2)),
        beta=float(reward.get("beta", 1.0)),
        gamma=float(reward.get("gamma", 1.0)),
        delta=float(reward.get("delta", 0.5)),
        dict_weight=float(reward.get("dict_weight", 0.5)),
        gt_weight=float(reward.get("gt_weight", 0.5)),
        min_format_words=int(reward.get("min_format_words", 8)),
    )


def keywords_match(
    rationale: str,
    gold_rationale: str | None,
    gold_category: str | None,
    category_keywords: dict[str, list[str]] | None,
    extra_exclude=None,
    dict_weight: float = 0.5,
    gt_weight: float = 0.5,
) -> dict[str, float]:
    denom = dict_weight + gt_weight
    if denom <= 0:
        return {"keywords": 0.0, "keywords_dict": 0.0, "keywords_gt": 0.0}

    lexicon = category_keywords if category_keywords is not None else DEFAULT_CATEGORY_KEYWORDS
    cat_key = (gold_category or "").strip().lower()
    dict_terms = lexicon.get(cat_key, [])
    dict_s = dictionary_score(rationale, dict_terms)

    exclude = list(extra_exclude or [])
    if gold_category:
        exclude.append(gold_category)
    gold_toks = content_tokens(gold_rationale or "", extra_exclude=exclude)
    pred_toks = content_tokens(rationale, extra_exclude=exclude)
    gt_s = token_f1(pred_toks, gold_toks)

    mixed = (dict_weight * dict_s + gt_weight * gt_s) / denom
    return {
        "keywords": mixed,
        "keywords_dict": dict_s,
        "keywords_gt": gt_s,
    }


def compute_reward(
    *,
    rationale: str,
    gold_rationale: str | None = None,
    gold_binary: int | None = None,
    pred_binary: int | None = None,
    gold_category: str | None = None,
    pred_category: str | None = None,
    score_category: bool = True,
    category_names=None,
    category_keywords: dict[str, list[str]] | None = None,
    weights: RewardWeights | None = None,
) -> dict[str, float]:
    """
    R = α·format + β·decision + γ·category + δ·keywords

    decision and category are 0/1 head correctness.
    category is 0 when that row should not score the K-way head.
    """
    w = weights or RewardWeights()
    fmt = format_score(rationale, category_names, min_words=w.min_format_words)
    decision = 0.0
    if gold_binary is not None and pred_binary is not None:
        decision = 1.0 if int(gold_binary) == int(pred_binary) else 0.0
    category = 0.0
    if score_category and gold_category and pred_category:
        category = (
            1.0
            if str(gold_category).strip().lower() == str(pred_category).strip().lower()
            else 0.0
        )
    kw = keywords_match(
        rationale,
        gold_rationale,
        gold_category,
        category_keywords,
        extra_exclude=category_names,
        dict_weight=w.dict_weight,
        gt_weight=w.gt_weight,
    )
    total = (
        w.alpha * fmt
        + w.beta * decision
        + w.gamma * category
        + w.delta * kw["keywords"]
    )
    return {
        "reward": total,
        "format": fmt,
        "decision": decision,
        "category": category,
        **kw,
        "alpha": w.alpha,
        "beta": w.beta,
        "gamma": w.gamma,
        "delta": w.delta,
    }
