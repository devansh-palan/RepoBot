"""Gold question set, retrieval metrics, and the ablation runner.

Metrics are pure functions over retrieved results, so the eval can score any
retriever without an LLM in the loop.
"""

from .gold import (
    GOLD_PATH,
    GoldAnswer,
    GoldQuestion,
    GoldSet,
    RepoRef,
    load_gold,
    summary,
    validate,
)
from .metrics import (
    hit_rate_at_k,
    matched_answers,
    matches,
    mean,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "GOLD_PATH",
    "GoldAnswer",
    "GoldQuestion",
    "GoldSet",
    "RepoRef",
    "hit_rate_at_k",
    "load_gold",
    "matched_answers",
    "matches",
    "mean",
    "recall_at_k",
    "reciprocal_rank",
    "summary",
    "validate",
]
