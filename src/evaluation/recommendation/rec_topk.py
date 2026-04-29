from __future__ import annotations

import numpy as np

from application.config import tags_26
from application.music_repository import load_tracks
from application.scoring import compute_scores, topk_indices
from application.user_state import UserStateEstimator

from common import (
    strategies,
    gamma,
    music_db_json,
    rec_out_dir,
    strength,
    top_k,
    fmt_pair,
    load_queries,
    save_csv,
)


def top5_tags(tag_probs):
    tag_probs = np.asarray(tag_probs, dtype=np.float32)
    idx = np.argsort(-tag_probs)[:5]
    return ", ".join(tags_26[int(i)] for i in idx)


def rank_topk(tracks, z_matrix, y_matrix, mu_q, tau):
    scores = compute_scores(z_matrix,y_matrix,mu_q,tau,gamma,)

    va_distance = scores["va_distance"]
    va_score = scores["va_score"]
    tag_score = scores["tag_score"]
    total_score = scores["total_score"]

    top_idx = topk_indices(total_score, top_k)

    rows = []
    for rank, i in enumerate(top_idx, start=1):
        track = tracks[int(i)]

        rows.append(
            {
                "Rank": rank,
                "Song ID": track["song_id"],
                "Song VA": fmt_pair(track["va"]),
                "Display Tags": ", ".join(track.get("display_tags", [])),
                "Top-5 Pred Tags": top5_tags(track["tag26"]),
                "VA distance ↓": float(va_distance[int(i)]),
                "VA score": float(va_score[int(i)]),
                "Tag score ↑": float(tag_score[int(i)]),
                "Total score ↑": float(total_score[int(i)]),
                "Song Path": track.get("song_path", ""),
            }
        )

    return rows


def main():
    queries = load_queries()
    tracks = load_tracks(music_db_json)
    estimator = UserStateEstimator()

    z_matrix = np.asarray([t["va"] for t in tracks], dtype=np.float32)
    y_matrix = np.asarray([t["tag26"] for t in tracks], dtype=np.float32)

    main_rows = []
    topk_rows = []

    for q in queries:
        z_q = estimator.infer_z_q(q["text"])

        for strategy in strategies:
            state = estimator.infer_mu_q(
                z_q=z_q,
                strategy=strategy,
                strength=strength,
            )

            top_rows = rank_topk(
                tracks=tracks,
                z_matrix=z_matrix,
                y_matrix=y_matrix,
                mu_q=state["mu_q"],
                tau=state["tag_prior"],
            )

            top1 = top_rows[0]
            query_id = f"{q['base_id']}_{strategy}"

            main_rows.append(
                {
                    "Query ID": query_id,
                    "Quadrant": q["quadrant"],
                    "Quadrant Name": q["quadrant_name"],
                    "Expected VA": q["expected_va"],
                    "Language": q["language"],
                    "Text": q["text"],
                    "Predicted z_q": fmt_pair(state["z_q"]),
                    "Strategy": strategy,
                    "Strength": state["strength"],
                    "Anchor": fmt_pair(state["anchor"]),
                    "Target μ_q": fmt_pair(state["mu_q"]),
                    "Delta": fmt_pair(state["delta"]),
                    "Top-1 Song": top1["Song ID"],
                    "Top-1 VA": top1["Song VA"],
                    "Top-5 Tags": top1["Top-5 Pred Tags"],
                    "VA distance ↓": top1["VA distance ↓"],
                    "Tag score ↑": top1["Tag score ↑"],
                    "Total score ↑": top1["Total score ↑"],
                    "Song Path": top1["Song Path"],
                }
            )

            for row in top_rows:
                topk_rows.append(
                    {
                        "Query ID": query_id,
                        "Quadrant": q["quadrant"],
                        "Language": q["language"],
                        "Text": q["text"],
                        "Strategy": strategy,
                        "Predicted z_q": fmt_pair(state["z_q"]),
                        "Target μ_q": fmt_pair(state["mu_q"]),
                        **row,
                    }
                )

            print(query_id, "->", top1["Song ID"], top1["Top-5 Pred Tags"])

    save_csv(main_rows, rec_out_dir / "rec_results.csv")
    save_csv(topk_rows, rec_out_dir / f"rec_top{top_k}.csv")

if __name__ == "__main__":
    main()
