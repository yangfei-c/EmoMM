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


def load_embd(folder):
    ids = []
    vecs = []

    # 不用 rglob，避免误读子目录；这里只读当前目录下的 .npy
    for p in sorted(folder.glob("*.npy")):
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


def build_track_info():
    """
    从全量音乐库读取 metadata。
    这里不能再用 candidate_pool，因为 CLaMP3 已经跑了全量 49553 首。
    """
    tracks = load_tracks()

    info = {}
    duplicate = 0

    for t in tracks:
        song_id = str(t.get("song_id", "")).strip()

        if not song_id:
            continue

        if song_id in info:
            duplicate += 1
            continue

        info[song_id] = t

    print(f"全量 metadata 数量: {len(info)}")
    print(f"重复 song_id 跳过数量: {duplicate}")

    return info


def main():
    queries = json.loads(QUERY_JSON.read_text(encoding="utf-8"))

    track_info = build_track_info()

    query_ids, query_emb = load_embd(QUERY_EMBD)
    song_ids, audio_emb = load_embd(AUDIO_EMBD)

    query_index = {qid: i for i, qid in enumerate(query_ids)}

    print(f"query embedding: {query_emb.shape}")
    print(f"audio embedding: {audio_emb.shape}")
    print(f"audio ids 数量: {len(song_ids)}")

    if query_emb.shape[1] != audio_emb.shape[1]:
        raise ValueError(f"维度不一致: query={query_emb.shape}, audio={audio_emb.shape}")

    missing_meta = [sid for sid in song_ids if sid not in track_info]
    print(f"audio embedding 中找不到 metadata 的数量: {len(missing_meta)}")

    if missing_meta:
        print("示例 missing metadata song_id:")
        for sid in missing_meta[:10]:
            print("  ", sid)

    rows = []

    for q in queries:
        qid = get_query_id(q)

        if qid not in query_index:
            raise KeyError(f"query_embd 里找不到: {qid}")

        scores = audio_emb @ query_emb[query_index[qid]]
        top_idx = np.argsort(-scores)[:TOP_K]

        for rank, idx in enumerate(top_idx, start=1):
            idx = int(idx)

            song_id = song_ids[idx]
            info = track_info.get(song_id, {})

            rows.append({
                "method": "clamp3",
                "query_id": qid,
                "rank": rank,
                "song_id": song_id,
                "score": round(float(scores[idx]), 6),
                "text": q.get("text", ""),
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

    expected_rows = len(queries) * TOP_K
    if len(rows) != expected_rows:
        print(f"Warning: 期望 {expected_rows} 行，实际 {len(rows)} 行")

    empty_audio_path = sum(1 for r in rows if not r["audio_path"])
    empty_va = sum(1 for r in rows if r["va"] in ["[]", ""])

    print(f"audio_path 为空数量: {empty_audio_path}")
    print(f"va 为空数量: {empty_va}")


if __name__ == "__main__":
    main()