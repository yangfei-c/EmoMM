from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence


def clip_value(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def clip_va(z: Sequence[float]) -> List[float]:
    return [clip_value(float(z[0])), clip_value(float(z[1]))]


def interpolate_va(z_q: Sequence[float], z_anchor: Sequence[float], strength: float) -> List[float]:
    """Target VA: mu_q = (1-s) * z_q + s * z_anchor."""
    s = max(0.0, min(1.0, float(strength)))
    v = (1.0 - s) * float(z_q[0]) + s * float(z_anchor[0])
    a = (1.0 - s) * float(z_q[1]) + s * float(z_anchor[1])
    return clip_va([v, a])


def make_tag_prior(labels: Sequence[str], tag26: Sequence[str]) -> List[float]:
    """Uniform 26-d prior over selected strategy labels."""
    prior = [0.0] * len(tag26)
    selected = [x for x in labels if x in tag26]
    if not selected:
        return prior

    w = 1.0 / len(selected)
    index = {tag: i for i, tag in enumerate(tag26)}
    for label in selected:
        prior[index[label]] = w
    return prior


def va_score(z_i: Sequence[float], mu_q: Sequence[float]) -> float:
    # Smaller distance => larger score.
    return -((float(z_i[0]) - float(mu_q[0])) ** 2 + (float(z_i[1]) - float(mu_q[1])) ** 2)


def tag_score(y_i: Sequence[float], tau: Sequence[float]) -> float:
    return sum(float(y) * float(t) for y, t in zip(y_i, tau))


def total_score(
    z_i: Sequence[float],
    y_i: Sequence[float],
    mu_q: Sequence[float],
    tau: Sequence[float],
    gamma: float = 1.0,
) -> float:
    return va_score(z_i, mu_q) + float(gamma) * tag_score(y_i, tau)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) ** 2 for x in a))
    nb = math.sqrt(sum(float(y) ** 2 for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def mmr_rerank(
    items: Sequence[Dict[str, object]],
    score_key: str,
    top_k: int,
    diversity_weight: float = 0.0,
    vector_key: str = "tag26",
) -> List[Dict[str, object]]:
    """MMR-style re-ranking using relevance score and tag-vector dissimilarity.

    The original score decides relevance. diversity_weight controls how much we penalize
    similarity to already selected tracks. This is a post-processing step and does not
    change the original VA/Tag scoring formula.
    """
    if not items or top_k <= 0:
        return []

    eta = max(0.0, min(1.0, float(diversity_weight)))
    pool = list(items)
    if eta <= 0.0:
        return sorted(pool, key=lambda x: float(x.get(score_key, 0.0)), reverse=True)[:top_k]

    base_scores = [float(item.get(score_key, 0.0)) for item in pool]
    min_s, max_s = min(base_scores), max(base_scores)
    span = max_s - min_s
    rel_scores = [1.0 if span <= 1e-12 else (s - min_s) / span for s in base_scores]

    selected: List[int] = []
    remaining = set(range(len(pool)))

    while remaining and len(selected) < top_k:
        best_idx = None
        best_value = -float("inf")

        for idx in remaining:
            if not selected:
                diversity_penalty = 0.0
            else:
                vec_i = pool[idx].get(vector_key, [])
                diversity_penalty = max(
                    cosine_similarity(vec_i, pool[j].get(vector_key, [])) for j in selected
                )

            value = (1.0 - eta) * rel_scores[idx] - eta * diversity_penalty
            if value > best_value:
                best_value = value
                best_idx = idx

        selected.append(best_idx)  # type: ignore[arg-type]
        remaining.remove(best_idx)  # type: ignore[arg-type]

    return [pool[i] for i in selected]


def load_music_db(path: Path) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else list(data.values())
