# -*- coding: utf-8 -*-

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from core.music_repository import load_tracks
from core.tag_prototype import compute_tag_prototypes
from core.utils import select_nearest_tags, make_prototype_prior
from core.scoring import compute_scores, topk_indices
from core.user_state import UserStateEstimator
from core.config import strategy_tag


QUERY_JSON = ROOT / "eval" / "queries_zh16.json"
OUT_CSV = ROOT / "eval" / "ours" / "ours_top5.csv"

TOP_K = 5

TAG26 = [
    "ambient", "classical", "electronic", "folk", "hiphop", "jazz",
    "metal", "pop", "rock", "soundtrack",
    "bass", "drums", "guitar", "piano", "strings", "synthesizer", "voice",
    "dark", "dream", "energetic", "epic", "film", "happy", "love", "relaxing", "sad",
]


def load_queries():
    return json.loads(QUERY_JSON.read_text(encoding="utf-8"))


def qid(q):
    return str(q.get("query_id") or q.get("id"))


def query_current_text(q):
    text = str(q.get("current_state") or "").strip()
    if not text:
        raise ValueError(f"{qid(q)} 缺少 current_state")
    return text


def query_full_text(q):
    text = str(q.get("text") or "").strip()
    if text:
        return text
    return f"{q.get('current_state', '')}{q.get('regulation_goal', '')}".strip()


def query_strategy(q):
    s = str(q.get("strategy") or "").strip()
    if not s:
        raise ValueError(f"{qid(q)} 缺少 strategy")
    return s


def query_labels(q, strategy):
    labels = q.get("strategy_tags")
    if labels is None:
        labels = strategy_tag.get(strategy, [])
    return labels


def load_music():
    tracks = [
        t for t in load_tracks()
        if t.get("va") is not None and t.get("tag26") is not None
    ]
    z_matrix = np.asarray([t["va"] for t in tracks], dtype=np.float32)
    y_matrix = np.asarray([t["tag26"] for t in tracks], dtype=np.float32)
    return tracks, z_matrix, y_matrix


def row_from_track(q, track, rank, score, z_q, mu_q):
    return {
        "method": "ours",
        "query_id": qid(q),
        "rank": rank,
        "song_id": str(track["song_id"]),
        "score": round(float(score), 6),

        "text": query_full_text(q),
        "current_state": str(q.get("current_state", "")).strip(),
        "regulation_goal": str(q.get("regulation_goal", "")).strip(),
        "strategy": str(q.get("strategy", "")).strip(),
        "strategy_zh": str(q.get("strategy_zh") or q.get("strategy") or "").strip(),
        "strength": float(q.get("strength", 0.5)),
        "z_q": json.dumps(z_q, ensure_ascii=False),
        "mu_q": json.dumps(mu_q, ensure_ascii=False),

        "display_tags": "|".join(map(str, track.get("display_tags", []))),
        "va": json.dumps(track.get("va", []), ensure_ascii=False),
        "audio_path": str(track.get("song_path") or track.get("audio_path") or ""),
    }


def write_csv(rows):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "method", "query_id", "rank", "song_id", "score",
        "text", "current_state", "regulation_goal",
        "strategy", "strategy_zh", "strength", "z_q", "mu_q",
        "display_tags", "va", "audio_path",
    ]

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    queries = load_queries()
    tracks, z_matrix, y_matrix = load_music()
    prototypes = compute_tag_prototypes(tracks)
    estimator = UserStateEstimator()

    rows = []

    for q in queries:
        strategy = query_strategy(q)
        strength = float(q.get("strength", 0.5))

        state = estimator.process(
            text=query_current_text(q),
            strategy=strategy,
            strength=strength,
        )

        z_q = state["z_q"]
        mu_q = state["mu_q"]

        labels = query_labels(q, strategy)
        if labels == ["auto"]:
            labels = select_nearest_tags(mu_q, prototypes, TAG26)

        tau = make_prototype_prior(
            mu_q=mu_q,
            labels=labels,
            tag26=TAG26,
            prototypes=prototypes,
            alpha=10.0,
        )

        scores = compute_scores(
            z_matrix=z_matrix,
            y_matrix=y_matrix,
            z_q=z_q,
            mu_q=mu_q,
            tau=tau,
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
        )["total_score"]

        for rank, idx in enumerate(topk_indices(scores, TOP_K), start=1):
            idx = int(idx)
            rows.append(row_from_track(q, tracks[idx], rank, scores[idx], z_q, mu_q))

    write_csv(rows)

    print(f"完成 Ours Top-{TOP_K}: {OUT_CSV}")
    print(f"输出行数: {len(rows)}")


if __name__ == "__main__":
    main()