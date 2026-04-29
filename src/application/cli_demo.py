from __future__ import annotations

import numpy as np

from application.config import default_strength
from application.music_repository import load_tracks
from application.scoring import compute_scores
from application.user_state import UserStateEstimator
from config import text_list


def build_candidates(mu_q, tau, gamma=1.0):
    tracks = load_tracks()

    z_matrix = np.asarray([t["va"] for t in tracks], dtype=np.float32)
    y_matrix = np.asarray([t["tag26"] for t in tracks], dtype=np.float32)

    scores = compute_scores(z_matrix,y_matrix,mu_q,tau,gamma)

    results = []
    for i, track in enumerate(tracks):
        z_i = track["va"]

        results.append({
            "song_id": track["song_id"],
            "song_path": track["song_path"],
            "display_tags": track["display_tags"],
            "va": z_i,
            "va_score": float(scores["va_score"][i]),
            "tag_score": float(scores["tag_score"][i]),
            "total_score": float(scores["total_score"][i]),
        })
    return results

def topk_by(results, key, top_k=5):
    return sorted(results, key=lambda x: float(x[key]), reverse=True)[:top_k]


def print_group(title, items):
    print("\n" + title)
    print("-" * 100)

    for rank, item in enumerate(items, start=1):
        print(f"[Top {rank}] {item['song_id']}")
        print(f"  VA: {item['va']}")
        print(f"  Display tags: {item['display_tags']}")
        print(f"  VA score: {item['va_score']:.4f}")
        print(f"  Tag score: {item['tag_score']:.4f}")
        print(f"  VA+Tag score: {item['total_score']:.4f}")
        print(f"  Path: {item['song_path']}")
        print()


def run_one(text, strategy, strength=default_strength, top_k=5, gamma=1.0):
    estimator = UserStateEstimator()
    state = estimator.process(text, strategy=strategy, strength=strength)

    mu_q = state["mu_q"]
    tau = state["tag_prior"]

    results = build_candidates(mu_q, tau, gamma=gamma)

    va_top = topk_by(results, "va_score", top_k=top_k)
    tag_top = topk_by(results, "tag_score", top_k=top_k)
    both_top = topk_by(results, "total_score", top_k=top_k)

    print("=" * 100)
    print(f"Text: {text}")
    print(f"Strategy: {strategy}")
    print(f"Strength: {state['strength']}")
    print(f"Current VA z_q: {state['z_q']}")
    print(f"Strategy anchor: {state['anchor']}")
    print(f"Target VA mu_q: {state['mu_q']}")
    print(f"Delta: {state['delta']}")

    print_group("1) 只看 VA：按目标 VA 距离排序", va_top)
    print_group("2) 只看 Tag：按策略标签匹配排序", tag_top)
    print_group("3) VA + Tag：最终联合排序", both_top)


def main():
    for item in text_list:
        run_one(
            text=item["text"],
            strategy=item["intent"],
            strength=default_strength,
            top_k=5,
            gamma=1.0,
        )


if __name__ == "__main__":
    main()
