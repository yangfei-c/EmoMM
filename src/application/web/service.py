from __future__ import annotations

import numpy as np
import streamlit as st

from application.config import (
    diversity_weights,
    rerank_pool_size,
    strategy_tag,
    tags_26,
    prototype_alpha,
    prototype_top_percent,
)
from application.music_repository import load_tracks
from application.scoring import compute_scores
from application.tag_prototype import compute_tag_prototypes
from application.utils import (
    make_uniform_prior,
    make_prototype_prior,
    select_nearest_tags,
    mmr_rerank,
)


@st.cache_resource(show_spinner="正在加载 Text VA 模型...")
def get_estimator():
    from application.user_state import UserStateEstimator
    return UserStateEstimator()


@st.cache_data(show_spinner="正在加载音乐库...")
def get_tracks():
    return load_tracks()


@st.cache_data(show_spinner="正在计算标签原型...")
def get_tag_prototypes():
    tracks = get_tracks()
    return compute_tag_prototypes(
        tracks,
        top_percent=prototype_top_percent,
    )


def build_tag_prior(strategy, mu_q, prior_mode="prototype"):
    """
    Build strategy tag prior.

    prior_mode:
        - uniform: manually selected strategy labels with equal weights
        - prototype: strategy labels reweighted by distance to tag prototypes
    """
    prototypes = get_tag_prototypes()
    labels = strategy_tag.get(strategy, [])

    if labels == "auto" or labels == ["auto"]:
        labels = select_nearest_tags(
            mu_q=mu_q,
            prototypes=prototypes,
            tag26=tags_26,
        )

    if prior_mode == "uniform":
        return make_uniform_prior(labels, tags_26), labels

    if prior_mode == "prototype":
        return make_prototype_prior(
            mu_q=mu_q,
            labels=labels,
            tag26=tags_26,
            prototypes=prototypes,
            alpha=prototype_alpha,
        ), labels

    raise ValueError(f"Unknown prior_mode: {prior_mode}")


def build_candidates(
    z_q,
    mu_q,
    tau,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
):
    tracks = get_tracks()

    z_matrix = np.asarray([t["va"] for t in tracks], dtype=np.float32)
    y_matrix = np.asarray([t["tag26"] for t in tracks], dtype=np.float32)

    scores = compute_scores(
        z_matrix=z_matrix,
        y_matrix=y_matrix,
        z_q=z_q,
        mu_q=mu_q,
        tau=tau,
        alpha=alpha,
        beta=beta,
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
            "curr_dist": float(scores["curr_dist"][i]),
            "target_dist": float(scores["target_dist"][i]),
            "curr_score": float(scores["curr_score"][i]),
            "target_score": float(scores["target_score"][i]),
            "va_score": float(scores["va_score"][i]),
            "tag_score": float(scores["tag_score"][i]),
            "total_score": float(scores["total_score"][i]),
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


def recommend(
    text,
    strategy,
    strength,
    gamma,
    top_k,
    diversity_mode="关闭",
    prior_mode="prototype",
):
    """
    Main recommendation service.

    The user-facing demo uses the final method by default:
    current-state smoothing + target VA matching + prototype-based tag prior.
    """
    estimator = get_estimator()

    state = estimator.process(
        text,
        strategy=strategy,
        strength=strength,
    )

    tau, selected_labels = build_tag_prior(
        strategy=strategy,
        mu_q=state["mu_q"],
        prior_mode=prior_mode,
    )

    state["tag_prior"] = tau
    state["selected_strategy_tags"] = selected_labels
    state["prior_mode"] = prior_mode

    results = build_candidates(
        z_q=state["z_q"],
        mu_q=state["mu_q"],
        tau=tau,
        alpha=1.0,
        beta=1.0,
        gamma=gamma,
    )

    eta = float(diversity_weights.get(diversity_mode, 0.0))

    ours = topk_by(results, "total_score", top_k, diversity_weight=eta)

    return state, {
        "Ours": ours,
        # Keep these keys for experiment scripts or later ablation analysis.
        "TargetVA-only": topk_by(results, "target_score", top_k, diversity_weight=eta),
        "VA-path": topk_by(results, "va_score", top_k, diversity_weight=eta),
        "Tag-only": topk_by(results, "tag_score", top_k, diversity_weight=eta),
        # Backward-compatible alias.
        "VA + Tag": ours,
    }, results, eta
