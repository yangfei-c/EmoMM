from __future__ import annotations
import numpy as np


def compute_scores(z_matrix,y_matrix,mu_q,tau,gamma: float = 1.0,):
    """All-track"""
    mu = np.asarray(mu_q, dtype=np.float32)
    tau = np.asarray(tau, dtype=np.float32)

    va_distance = np.sum((z_matrix - mu[None, :]) ** 2, axis=1)
    va_score = -va_distance
    tag_score = y_matrix @ tau
    total_score = va_score + float(gamma) * tag_score

    return {
        "va_distance": va_distance,
        "va_score": va_score,
        "tag_score": tag_score,
        "total_score": total_score,
    }


def get_method_score(method: str,scores: dict):
    if method == "VA-only":
        return scores["va_score"]
    if method == "Tag-only":
        return scores["tag_score"]
    if method in {"Proposed VA+Tag", "VA+Tag", "VA + Tag"}:
        return scores["total_score"]
    raise ValueError(f"Unknown method: {method}")

def topk_indices(score, top_k: int):
    return np.argsort(-np.asarray(score))[:top_k]
