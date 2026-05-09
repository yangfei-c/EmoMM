# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path

import numpy as np
from msclap import CLAP


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

QUERY_JSON = ROOT / "eval" / "queries_zh16.json"
POOL_JSON = ROOT / "eval" / "candidate_pool.json"

CLAP_DIR = Path(r"D:\AAAA_audio_dataset\eval\clap")
AUDIO_EMB = CLAP_DIR / "audio_embd.npz"
TEXT_EMB = CLAP_DIR / "text_embd.npz"

OUT_CSV = ROOT / "eval" / "clap" / "clap_top5.csv"

TOP_K = 5
BATCH_SIZE = 16


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def get_query_id(q):
    return str(q.get("query_id") or q.get("id"))


def main():
    CLAP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    queries = json.loads(QUERY_JSON.read_text(encoding="utf-8"))
    pool = json.loads(POOL_JSON.read_text(encoding="utf-8"))

    pool_info = {str(x["song_id"]): x for x in pool}

    query_ids = [get_query_id(q) for q in queries]
    query_texts = [q["text"] for q in queries]

    song_ids = [str(x["song_id"]) for x in pool]
    audio_paths = [str(x["song_path"]) for x in pool]

    print(f"query 数量: {len(query_texts)}")
    print(f"候选歌曲数量: {len(audio_paths)}")

    model = CLAP(version="2023", use_cuda=True)

    if AUDIO_EMB.exists():
        data = np.load(AUDIO_EMB, allow_pickle=True)
        song_ids = [str(x) for x in data["song_ids"]]
        audio_paths = [str(x) for x in data["audio_paths"]]
        audio_emb = data["audio_emb"].astype(np.float32)
        print(f"读取 audio embedding: {AUDIO_EMB}")
    else:
        audio_emb_list = []

        for i in range(0, len(audio_paths), BATCH_SIZE):
            batch = audio_paths[i:i + BATCH_SIZE]
            emb = model.get_audio_embeddings(batch)
            audio_emb_list.append(to_numpy(emb).astype(np.float32))
            print(f"audio embedding: {min(i + BATCH_SIZE, len(audio_paths))}/{len(audio_paths)}")

        audio_emb = np.concatenate(audio_emb_list, axis=0)

        np.savez(
            AUDIO_EMB,
            song_ids=np.array(song_ids),
            audio_paths=np.array(audio_paths),
            audio_emb=audio_emb,
        )
        print(f"保存 audio embedding: {AUDIO_EMB}")

    text_emb_list = []

    for i, text in enumerate(query_texts):
        emb = model.get_text_embeddings([text])
        text_emb_list.append(to_numpy(emb).astype(np.float32))
        print(f"text embedding: {i + 1}/{len(query_texts)}")

    text_emb = np.concatenate(text_emb_list, axis=0)

    np.savez(
        TEXT_EMB,
        query_ids=np.array(query_ids),
        query_texts=np.array(query_texts),
        text_emb=text_emb,
    )
    print(f"保存 text embedding: {TEXT_EMB}")

    scores = norm(audio_emb) @ norm(text_emb).T

    rows = []

    for qi, q in enumerate(queries):
        top_idx = np.argsort(-scores[:, qi])[:TOP_K]

        for rank, idx in enumerate(top_idx, start=1):
            song_id = song_ids[int(idx)]
            info = pool_info.get(song_id, {})

            rows.append({
                "method": "clap",
                "query_id": query_ids[qi],
                "rank": rank,
                "song_id": song_id,
                "score": round(float(scores[int(idx), qi]), 6),
                "text": q["text"],
                "strategy": q.get("strategy", ""),
                "display_tags": "|".join(map(str, info.get("display_tags", []))),
                "va": json.dumps(info.get("va", []), ensure_ascii=False),
                "audio_path": audio_paths[int(idx)],
            })

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