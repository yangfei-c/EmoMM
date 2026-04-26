from __future__ import annotations

import streamlit as st

from application.config import diversity_weights, rerank_pool_size
from application.music_repository import load_tracks
from application.utils import mmr_rerank, tag_score, total_score, va_score


@st.cache_resource(show_spinner="正在加载 Text VA 模型...")
def get_estimator():
    from application.user_state import UserStateEstimator

    return UserStateEstimator()


@st.cache_data(show_spinner="正在加载音乐库...")
def get_tracks():
    return load_tracks()


def build_candidates(mu_q, tau, gamma=1.0):
    results = []

    for track in get_tracks():
        z_i = track["va"]
        y_i = track["tag26"]

        v_score = va_score(z_i, mu_q)
        t_score = tag_score(y_i, tau)
        both_score = total_score(z_i, y_i, mu_q, tau, gamma=gamma)

        results.append({
            "song_id": track["song_id"],
            "song_path": track["song_path"],
            "display_tags": ", ".join(track["display_tags"]),
            "tag26": y_i,
            "va": z_i,
            "V": float(z_i[0]),
            "A": float(z_i[1]),
            "va_score": float(v_score),
            "tag_score": float(t_score),
            "both_score": float(both_score),
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
