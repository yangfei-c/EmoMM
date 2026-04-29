from __future__ import annotations

import numpy as np
import pandas as pd

from cfg_metrics import (
    OUT_DIR,
    TOP_K,
    STRENGTH,
    METHODS,
    EXPERIMENT_STRATEGIES,
    RANDOM_SEED,
    fmt_pair,
    load_queries,
    score_tracks,
    MUSIC_DB_JSON
)

from application.music_repository import load_tracks
from application.user_state import UserStateEstimator


def main():
    queries = load_queries()
    tracks = load_tracks(MUSIC_DB_JSON)
    estimator = UserStateEstimator()
    rng = np.random.default_rng(RANDOM_SEED)

    z_matrix = np.asarray([t["va"] for t in tracks], dtype=np.float32)
    y_matrix = np.asarray([t["tag26"] for t in tracks], dtype=np.float32)

    rows = []

    for q in queries:
        z_q = estimator.infer_z_q(q["text"])

        for strategy in EXPERIMENT_STRATEGIES:
            state = estimator.infer_mu_q(
                z_q=z_q,
                strategy=strategy,
                strength=STRENGTH,
            )

            query_id = f"{q['base_id']}_{strategy}"

            for method in METHODS:
                score, va_distance, tag_score = score_tracks(
                    method=method,
                    z_matrix=z_matrix,
                    y_matrix=y_matrix,
                    mu_q=state["mu_q"],
                    tau=state["tag_prior"],
                    rng=rng,
                )

                top_idx = np.argsort(-score)[:TOP_K]

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
                            "VA distance": float(va_distance[i]),
                            "Tag score": float(tag_score[i]),
                            "Song Path": track.get("song_path", ""),
                        }
                    )

            print(f"done: {query_id}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUT_DIR / "metric_topk_results.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()