# -*- coding: utf-8 -*-

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")
sys.path.insert(0, str(ROOT / "src"))

from core.music_repository import load_tracks
from core.tag_prototype import compute_tag_prototypes
from core.utils import select_nearest_tags, make_prototype_prior
from core.scoring import compute_scores, topk_indices


QUERY_JSON = ROOT / "eval" / "queries_zh16.json"
POOL_JSON = ROOT / "eval" / "candidate_pool.json"
OUT_CSV = ROOT / "eval" / "ours" / "ours_top5.csv"

TOP_K = 5

TAG26 = [
    "ambient", "classical", "electronic", "folk", "hiphop", "jazz",
    "metal", "pop", "rock", "soundtrack",
    "bass", "drums", "guitar", "piano", "strings", "synthesizer", "voice",
    "dark", "dream", "energetic", "epic", "film", "happy", "love", "relaxing", "sad",
]


def get_query_id(q):
    return str(q.get("query_id") or q.get("id"))


def main():
    queries = json.loads(QUERY_JSON.read_text(encoding="utf-8"))
    pool = json.loads(POOL_JSON.read_text(encoding="utf-8"))

    all_tracks = load_tracks()
    track_by_id = {str(t["song_id"]): t for t in all_tracks}

    tracks = []

    for item in pool:
        song_id = str(item["song_id"])
        if song_id not in track_by_id:
            raise KeyError(f"candidate_pool 中的 song_id 不在 load_tracks(): {song_id}")

        track = dict(track_by_id[song_id])
        track["song_path"] = item.get("song_path", track.get("song_path", ""))
        track["display_tags"] = item.get("display_tags", track.get("display_tags", []))
        track["va"] = item.get("va", track.get("va", []))
        tracks.append(track)

    if not tracks:
        raise RuntimeError("候选池为空。")

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
                "query_id": get_query_id(q),
                "rank": rank,
                "song_id": str(track["song_id"]),
                "score": round(float(scores["total_score"][idx]), 6),
                "text": q["text"],
                "strategy": q.get("strategy", ""),
                "display_tags": "|".join(map(str, track.get("display_tags", []))),
                "va": json.dumps(track.get("va", []), ensure_ascii=False),
                "audio_path": str(track.get("song_path", "")),
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
                "display_tags",
                "va",
                "audio_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Ours Top-5 生成完成")
    print(f"候选池: {POOL_JSON}")
    print(f"输出文件: {OUT_CSV}")
    print(f"query 数量: {len(queries)}")
    print(f"候选歌曲数量: {len(tracks)}")
    print(f"输出行数: {len(rows)}")


if __name__ == "__main__":
    main()