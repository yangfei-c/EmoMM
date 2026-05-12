# -*- coding: utf-8 -*-

import csv
import json
import sys
from pathlib import Path

import numpy as np
from msclap import CLAP


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")
sys.path.insert(0, str(ROOT / "src"))

from core.music_repository import load_tracks


QUERY_JSON = ROOT / "eval" / "queries_zh16.json"

CLAP_DIR = Path(r"D:\AAAA_audio_dataset\eval\clap")
AUDIO_EMB = CLAP_DIR / "audio_embd.npz"
TEXT_EMB = CLAP_DIR / "text_embd.npz"

OUT_CSV = ROOT / "eval" / "clap" / "clap_top5.csv"

TOP_K = 5


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
    if not AUDIO_EMB.exists():
        raise FileNotFoundError(f"找不到 CLAP audio embedding: {AUDIO_EMB}")

    queries = json.loads(QUERY_JSON.read_text(encoding="utf-8"))

    all_tracks = load_tracks()
    track_by_id = {str(t["song_id"]): t for t in all_tracks}

    audio_data = np.load(AUDIO_EMB, allow_pickle=True)

    song_ids = [str(x) for x in audio_data["song_ids"]]
    audio_paths = [str(x) for x in audio_data["audio_paths"]]
    audio_emb = audio_data["audio_emb"].astype(np.float32)

    print(f"读取 CLAP audio embedding: {AUDIO_EMB}")
    print(f"音频 embedding 数量: {len(song_ids)}")

    query_ids = [get_query_id(q) for q in queries]
    query_texts = [q["text"] for q in queries]

    model = CLAP(version="2023", use_cuda=True)

    if TEXT_EMB.exists():
        text_data = np.load(TEXT_EMB, allow_pickle=True)
        old_query_ids = [str(x) for x in text_data["query_ids"]]

        if old_query_ids == query_ids:
            text_emb = text_data["text_emb"].astype(np.float32)
            print(f"读取已有 text embedding: {TEXT_EMB}")
        else:
            text_emb = build_text_emb(model, query_ids, query_texts)
    else:
        text_emb = build_text_emb(model, query_ids, query_texts)

    scores = norm(audio_emb) @ norm(text_emb).T

    rows = []

    for qi, q in enumerate(queries):
        top_idx = np.argsort(-scores[:, qi])[:TOP_K]

        for rank, idx in enumerate(top_idx, start=1):
            idx = int(idx)

            song_id = song_ids[idx]
            track = track_by_id.get(song_id, {})

            rows.append({
                "method": "clap",
                "query_id": query_ids[qi],
                "rank": rank,
                "song_id": song_id,
                "score": round(float(scores[idx, qi]), 6),
                "text": q["text"],
                "strategy": q.get("strategy", ""),
                "display_tags": "|".join(map(str, track.get("display_tags", []))),
                "va": json.dumps(track.get("va", []), ensure_ascii=False),
                "audio_path": audio_paths[idx],
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "query_id", "rank", "song_id", "score",
                "text", "strategy", "display_tags", "va", "audio_path"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("CLAP full-library Top-5 生成完成")
    print(f"输出: {OUT_CSV}")
    print(f"输出行数: {len(rows)}")


def build_text_emb(model, query_ids, query_texts):
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

    return text_emb


if __name__ == "__main__":
    main()