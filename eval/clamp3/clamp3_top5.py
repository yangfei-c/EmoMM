# -*- coding: utf-8 -*-

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")
sys.path.insert(0, str(ROOT / "src"))

from core.music_repository import load_tracks


QUERY_JSON = ROOT / "eval" / "queries_zh16.json"

BASE_DIR = Path(r"D:\AAAA_audio_dataset\eval\clamp3")
QUERY_EMBD = BASE_DIR / "query_embd"
AUDIO_EMBD = BASE_DIR / "audio_embd"

OUT_CSV = ROOT / "eval" / "clamp3" / "clamp3_top5.csv"

TOP_K = 5


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def load_queries():
    return json.loads(QUERY_JSON.read_text(encoding="utf-8"))


def qid(q):
    return str(q.get("query_id") or q.get("id"))


def query_text(q):
    text = str(q.get("text") or "").strip()
    if text:
        return text

    text = f"{q.get('current_state', '')}{q.get('regulation_goal', '')}".strip()
    if not text:
        raise ValueError(f"{qid(q)} 缺少 text")
    return text


def load_embd(folder):
    ids = []
    vecs = []

    for p in sorted(folder.glob("*.npy")):
        x = np.load(p, allow_pickle=True).squeeze().astype(np.float32)

        if x.ndim != 1:
            raise ValueError(f"不是全局向量: {p}, shape={x.shape}")

        ids.append(p.stem)
        vecs.append(x)

    if not vecs:
        raise RuntimeError(f"没有读取到 embedding: {folder}")

    return ids, norm(np.stack(vecs))


def load_track_meta():
    meta = {}

    for t in load_tracks():
        song_id = str(t.get("song_id", "")).strip()
        if song_id and song_id not in meta:
            meta[song_id] = t

    return meta


def validate_query_embd(queries, query_ids):
    missing = [qid(q) for q in queries if qid(q) not in query_ids]

    if missing:
        raise RuntimeError(f"query_embd 缺少: {missing}")


def make_row(q, song_id, score, rank, track):
    return {
        "method": "clamp3",
        "query_id": qid(q),
        "rank": rank,
        "song_id": song_id,
        "score": round(float(score), 6),

        "text": query_text(q),
        "current_state": str(q.get("current_state", "")).strip(),
        "regulation_goal": str(q.get("regulation_goal", "")).strip(),
        "strategy": q.get("strategy", ""),
        "strategy_zh": str(q.get("strategy_zh") or q.get("strategy") or "").strip(),
        "strength": q.get("strength", 0.5),

        "display_tags": "|".join(map(str, track.get("display_tags", []))),
        "va": json.dumps(track.get("va", []), ensure_ascii=False),
        "audio_path": str(track.get("song_path") or track.get("audio_path") or ""),
    }


def write_csv(rows):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "method", "query_id", "rank", "song_id", "score",
        "text", "current_state", "regulation_goal",
        "strategy", "strategy_zh", "strength",
        "display_tags", "va", "audio_path",
    ]

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    queries = load_queries()

    query_ids, query_emb = load_embd(QUERY_EMBD)
    song_ids, audio_emb = load_embd(AUDIO_EMBD)
    tracks = load_track_meta()

    validate_query_embd(queries, query_ids)

    query_index = {x: i for i, x in enumerate(query_ids)}
    rows = []

    for q in queries:
        q_index = query_index[qid(q)]
        scores = audio_emb @ query_emb[q_index]
        top_idx = np.argsort(-scores)[:TOP_K]

        for rank, idx in enumerate(top_idx, start=1):
            idx = int(idx)
            song_id = song_ids[idx]
            rows.append(
                make_row(
                    q=q,
                    song_id=song_id,
                    score=scores[idx],
                    rank=rank,
                    track=tracks.get(song_id, {}),
                )
            )

    write_csv(rows)

    print(f"完成 CLaMP3 Top-{TOP_K}: {OUT_CSV}")
    print(f"输出行数: {len(rows)}")


if __name__ == "__main__":
    main()