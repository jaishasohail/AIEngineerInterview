"""Reciprocal rank fusion for lexical and dense candidate lists."""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[int]], *, k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        unique: list[int] = []
        seen: set[int] = set()
        for item_id in ranking:
            if item_id in seen:
                continue
            seen.add(item_id)
            unique.append(item_id)
        for rank, item_id in enumerate(unique, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
