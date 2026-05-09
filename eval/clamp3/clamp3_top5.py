# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

QUERY_JSON = ROOT / "eval" / "queries_zh16.json"
POOL_JSON = ROOT / "eval" / "candidate_pool.json"

BASE_DIR = Path(r"D:\AAAA_audio_dataset\eval\clamp3")
QUERY_EMBD = BASE_DIR / "query_embd"
AUDIO_EMBD = BASE_DIR / "audio_embd"

OUT_CSV = ROOT / "eval" / "clamp3" / "clamp3_top5.csv"

TOP_K = 5


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def load_embd(folder):
    ids = []
    vecs = []

    for p in sorted(folder.rglob("*.npy")):
        x = np.load(p, allow_pickle=True).squeeze().astype(np.float32)

        if x.ndim != 1:
            raise ValueError(f"不是全局向量: {p}, shape={x.shape}")

        ids.append(p.stem)
        vecs.append(x)

    if not vecs:
        raise RuntimeError(f"没有读取到 embedding: {folder}")

    return ids, norm(np.stack(vecs))


def get_query_id(q):
    return str(q.get("query_id") or q.get("id"))


def main():
    queries = json.loads(QUERY_JSON.read_text(encoding="utf-8"))
    pool = json.loads(POOL_JSON.read_text(encoding="utf-8"))

    pool_info = {str(x["song_id"]): x for x in pool}

    query_ids, query_emb = load_embd(QUERY_EMBD)
    song_ids, audio_emb = load_embd(AUDIO_EMBD)

    query_index = {qid: i for i, qid in enumerate(query_ids)}

    print(f"query embedding: {query_emb.shape}")
    print(f"audio embedding: {audio_emb.shape}")

    if query_emb.shape[1] != audio_emb.shape[1]:
        raise ValueError(f"维度不一致: query={query_emb.shape}, audio={audio_emb.shape}")

    rows = []

    for q in queries:
        qid = get_query_id(q)

        if qid not in query_index:
            raise KeyError(f"query_embd 里找不到: {qid}")

        scores = audio_emb @ query_emb[query_index[qid]]
        top_idx = np.argsort(-scores)[:TOP_K]

        for rank, idx in enumerate(top_idx, start=1):
            song_id = song_ids[int(idx)]
            info = pool_info.get(song_id, {})

            rows.append({
                "method": "clamp3",
                "query_id": qid,
                "rank": rank,
                "song_id": song_id,
                "score": round(float(scores[int(idx)]), 6),
                "text": q["text"],
                "strategy": q.get("strategy", ""),
                "display_tags": "|".join(map(str, info.get("display_tags", []))),
                "va": json.dumps(info.get("va", []), ensure_ascii=False),
                "audio_path": str(info.get("song_path", "")),
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

    print(f"完成: {OUT_CSV}")
    print(f"输出行数: {len(rows)}")


if __name__ == "__main__":
    main()