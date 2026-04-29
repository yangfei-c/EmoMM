from __future__ import annotations

import numpy as np

from application.music_repository import load_tracks
from application.scoring import compute_scores, get_method_score, topk_indices
from application.user_state import UserStateEstimator
from common import (
    strategies,
    gamma,
    methods,
    metric_out_dir,
    music_db_json,
    random_seed,
    strength,
    top_k,
    fmt_pair,
    load_queries,
    save_csv,
)


def main():
    queries = load_queries()
    tracks = load_tracks(music_db_json)
    estimator = UserStateEstimator()
    rng = np.random.default_rng(random_seed)

    z_matrix = np.asarray([t["va"] for t in tracks], dtype=np.float32)
    y_matrix = np.asarray([t["tag26"] for t in tracks], dtype=np.float32)

    rows = []

    for q in queries:
        z_q = estimator.infer_z_q(q["text"])

        for strategy in strategies:
            state = estimator.infer_mu_q(
                z_q=z_q,
                strategy=strategy,
                strength=strength,
            )

            query_id = f"{q['base_id']}_{strategy}"

            scores = compute_scores(
                z_matrix=z_matrix,
                y_matrix=y_matrix,
                mu_q=state["mu_q"],
                tau=state["tag_prior"],
                gamma=gamma,
            )

            for method in methods:
                score = get_method_score(
                    method,
                    scores,
                )

                top_idx = topk_indices(score, top_k)

                for rank, idx in enumerate(top_idx, start=1):
                    i = int(idx)
                    track = tracks[i]

                    rows.append(
                        {
                            "Query ID": query_id,
                            "Base ID": q["base_id"],
                            "Quadrant": q["quadrant"],
                            "Quadrant Name": q["quadrant_name"],
                            "Language": q["language"],
                            "Text": q["text"],
                            "Strategy": strategy,
                            "Method": method,
                            "Predicted z_q": fmt_pair(state["z_q"]),
                            "Target μ_q": fmt_pair(state["mu_q"]),
                            "Rank": rank,
                            "Song ID": track["song_id"],
                            "Song VA": fmt_pair(track["va"]),
                            "VA distance": float(scores["va_distance"][i]),
                            "Tag score": float(scores["tag_score"][i]),
                            "Song Path": track.get("song_path", ""),
                        }
                    )

            print(f"done: {query_id}")

    save_csv(rows, metric_out_dir / "metric_topk_results.csv")


if __name__ == "__main__":
    main()
