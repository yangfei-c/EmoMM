import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
QUERY_JSON = ROOT / "data" / "query" / "queries_zh16.json"
OUT_CSV = ROOT / "data" / "eval" / "ours" / "ours_top5.csv"
TOP_K = 5
TAG26 = [
    "ambient", "classical", "electronic", "folk", "hiphop", "jazz",
    "metal", "pop", "rock", "soundtrack",
    "bass", "drums", "guitar", "piano", "strings", "synthesizer", "voice",
    "dark", "dream", "energetic", "epic", "film", "happy", "love", "relaxing", "sad",
]

sys.path.insert(0, str(ROOT / "src"))

from core.music_repository import load_tracks
from core.tag_prototype import compute_tag_prototypes
from core.utils import select_nearest_tags, make_prototype_prior
from core.scoring import compute_scores, topk_indices

def main():
    if not QUERY_JSON.exists():
        raise FileNotFoundError(f"找不到 query 文件：{QUERY_JSON}")

    queries = json.loads(QUERY_JSON.read_text(encoding="utf-8"))
    tracks = load_tracks()

    if not tracks:
        raise RuntimeError("load_tracks() 没有读取到任何歌曲，请检查 core.config.music_db_path。")

    z_matrix = np.asarray([t["va"] for t in tracks], dtype=np.float32)
    y_matrix = np.asarray([t["tag26"] for t in tracks], dtype=np.float32)
    prototypes = compute_tag_prototypes(tracks)

    rows = []

    for q in queries:
        labels = q.get("strategy_tags", [])

        if labels == ["auto"]:
            labels = select_nearest_tags(q["mu_q"], prototypes, TAG26)

        tau = make_prototype_prior(
            mu_q=q["mu_q"],
            labels=labels,
            tag26=TAG26,
            prototypes=prototypes,
            alpha=10.0,
        )

        scores = compute_scores(
            z_matrix=z_matrix,
            y_matrix=y_matrix,
            z_q=q["z_q"],
            mu_q=q["mu_q"],
            tau=tau,
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
        )

        top_idx = topk_indices(scores["total_score"], TOP_K)

        for rank, idx in enumerate(top_idx, start=1):
            track = tracks[int(idx)]

            rows.append({
                "method": "ours",
                "query_id": q["query_id"],
                "rank": rank,
                "song_id": track["song_id"],
                "score": round(float(scores["total_score"][idx]), 6),
                "text": q["text"],
                "strategy": q["strategy"],
                "song_path": track.get("song_path", ""),
                "display_tags": "|".join(map(str, track.get("display_tags", []))),
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "query_id",
                "rank",
                "song_id",
                "score",
                "text",
                "strategy",
                "song_path",
                "display_tags",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Ours Top-5 生成完成")
    print(f"输入 query: {QUERY_JSON}")
    print(f"输出文件: {OUT_CSV}")
    print(f"query 数量: {len(queries)}")
    print(f"输出行数: {len(rows)}")


if __name__ == "__main__":
    main()