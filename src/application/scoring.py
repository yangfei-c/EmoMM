from __future__ import annotations

import numpy as np


def compute_scores(
    z_matrix,
    y_matrix,
    z_q,
    mu_q,
    tau,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
):
    """
    All-track scoring.

    target_score = -||z_i - mu_q||^2
    curr_score   = -||z_i - z_q||^2
    va_score     = alpha * curr_score + beta * target_score
    tag_score    = y_i^T tau
    total_score  = va_score + gamma * tag_score
    """
    target_va = np.asarray(mu_q, dtype=np.float32)
    curr_va = np.asarray(z_q, dtype=np.float32)
    tau = np.asarray(tau, dtype=np.float32)

    curr_dist = np.sum((z_matrix - curr_va[None, :]) ** 2, axis=1)
    target_dist = np.sum((z_matrix - target_va[None, :]) ** 2, axis=1)

    curr_score = -curr_dist
    target_score = -target_dist
    tag_score = y_matrix @ tau

    va_score = float(alpha) * curr_score + float(beta) * target_score
    total_score = va_score + float(gamma) * tag_score

    return {
        "curr_dist": curr_dist,
        "target_dist": target_dist,
        "curr_score": curr_score,
        "target_score": target_score,
        "va_score": va_score,
        "tag_score": tag_score,
        "total_score": total_score,
    }


def get_method_score(method: str, scores: dict):
    if method == "TargetVA-only":
        return scores["target_score"]
    if method == "VA-path":
        return scores["va_score"]
    if method == "Tag-only":
        return scores["tag_score"]
    if method in {"Ours", "VA + Tag", "VA+Tag", "Proposed"}:
        return scores["total_score"]
    raise ValueError(f"Unknown method: {method}")


def topk_indices(score, top_k: int):
    return np.argsort(-np.asarray(score))[:top_k]
