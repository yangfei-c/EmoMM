from __future__ import annotations

import numpy as np
import streamlit as st

from application.config import diversity_weights, rerank_pool_size
from application.music_repository import load_tracks
from application.scoring import compute_scores
from application.utils import mmr_rerank


@st.cache_resource(show_spinner="正在加载 Text VA 模型...")
def get_estimator():
    from application.user_state import UserStateEstimator

    return UserStateEstimator()


@st.cache_data(show_spinner="正在加载音乐库...")
def get_tracks():
    return load_tracks()


def build_candidates(mu_q, tau, gamma=1.0):
    tracks = get_tracks()

    z_matrix = np.asarray([t["va"] for t in tracks], dtype=np.float32)
    y_matrix = np.asarray([t["tag26"] for t in tracks], dtype=np.float32)

    scores = compute_scores(
        z_matrix=z_matrix,
        y_matrix=y_matrix,
        mu_q=mu_q,
        tau=tau,
        gamma=gamma,
    )

    results = []
    for i, track in enumerate(tracks):
        z_i = track["va"]
        y_i = track["tag26"]

        results.append({
            "song_id": track["song_id"],
            "song_path": track["song_path"],
            "display_tags": ", ".join(track["display_tags"]),
            "tag26": y_i,
            "va": z_i,
            "V": float(z_i[0]),
            "A": float(z_i[1]),
            "va_score": float(scores["va_score"][i]),
            "tag_score": float(scores["tag_score"][i]),
            "both_score": float(scores["total_score"][i]),
        })

    return results


def topk_by(results, key, top_k=5, diversity_weight=0.0):
    ranked = sorted(results, key=lambda x: float(x[key]), reverse=True)
    if diversity_weight <= 0.0:
        return ranked[:top_k]

    pool_size = min(len(ranked), max(rerank_pool_size, top_k))
    return mmr_rerank(
        ranked[:pool_size],
        score_key=key,
        top_k=top_k,
        diversity_weight=diversity_weight,
        vector_key="tag26",
    )


def recommend(text, strategy, strength, gamma, top_k, diversity_mode="低"):
    estimator = get_estimator()
    state = estimator.process(text, strategy=strategy, strength=strength)

    results = build_candidates(state["mu_q"], state["tag_prior"], gamma=gamma)
    eta = float(diversity_weights.get(diversity_mode, 0.0))

    return state, {
        "VA-only": topk_by(results, "va_score", top_k, diversity_weight=eta),
        "Tag-only": topk_by(results, "tag_score", top_k, diversity_weight=eta),
        "VA + Tag": topk_by(results, "both_score", top_k, diversity_weight=eta),
    }, results, eta
