# -*- coding: utf-8 -*-
"""
rank_clamp3_from_embeddings.py

作用：
读取 CLaMP3 已经提取好的文本和音频 embedding，计算余弦相似度，生成 CLaMP3 Top-5 推荐。

需要你先用 CLaMP3 提取：
1. query prompt embedding
2. music audio embedding

默认读取：
src/eval_experiment/clamp3_outputs/text_query_embeds/
src/eval_experiment/clamp3_outputs/music_audio_embeds/

输出：
src/eval_experiment/outputs/rankings/clamp3_top5.json
src/eval_experiment/outputs/rankings/clamp3_top5.csv

重要：
CLaMP3 输出的 .npy 文件命名可能因版本不同略有差异。
本脚本会尝试用 query_id 和 audio filename stem 匹配。
如果匹配失败，把输出目录结构发给我，我再帮你改映射。
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

QUERY_PATH = ROOT / "src" / "eval_experiment" / "data" / "query_test_32.json"
MUSIC_DB_PATH = ROOT / "src" / "eval_experiment" / "data" / "music_eval_db.json"

TEXT_EMBED_DIR = ROOT / "src" / "eval_experiment" / "clamp3_outputs" / "text_query_embeds"
AUDIO_EMBED_DIR = ROOT / "src" / "eval_experiment" / "clamp3_outputs" / "music_audio_embeds"

OUTPUT_DIR = ROOT / "src" / "eval_experiment" / "outputs" / "rankings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(x) + 1e-12
    return x / norm


def load_npy_vector(path: Path) -> np.ndarray:
    arr = np.load(path)
    return normalize_vec(arr)


def collect_npy_files(root: Path) -> Dict[str, Path]:
    """
    返回 {stem: path}。
    如果 CLaMP3 输出为 query_id.npy 或 484375.npy，这里可以直接匹配。
    """
    files = list(root.rglob("*.npy"))
    mapping = {}
    for p in files:
        mapping[p.stem] = p
    return mapping


def audio_stem_from_song_path(song_path: str) -> str:
    return Path(song_path).stem


def possible_audio_keys(song: Dict[str, Any]) -> List[str]:
    """
    尝试多种可能 key：
    - 原始 mp3 文件名，例如 484375
    - song_id，例如 track_0484375
    - song_id 去掉 track_0 前缀后的数字形式
    """
    keys = []
    song_id = song.get("song_id", "")
    audio_path = song.get("audio_path", "")

    if audio_path:
        keys.append(audio_stem_from_song_path(audio_path))

    if song_id:
        keys.append(song_id)
        keys.append(song_id.replace("track_", ""))
        keys.append(song_id.replace("track_0", ""))

    # 去重保持顺序
    seen = set()
    out = []
    for k in keys:
        if k and k not in seen:
            out.append(k)
            seen.add(k)
    return out


def main() -> None:
    queries = load_json(QUERY_PATH)
    songs = load_json(MUSIC_DB_PATH)

    text_files = collect_npy_files(TEXT_EMBED_DIR)
    audio_files = collect_npy_files(AUDIO_EMBED_DIR)

    print(f"text embedding files: {len(text_files)}")
    print(f"audio embedding files: {len(audio_files)}")

    if len(text_files) == 0:
        raise FileNotFoundError(f"没有找到文本 embedding .npy: {TEXT_EMBED_DIR}")
    if len(audio_files) == 0:
        raise FileNotFoundError(f"没有找到音频 embedding .npy: {AUDIO_EMBED_DIR}")

    # 加载文本 embedding
    query_embeds = {}
    missing_queries = []

    for q in queries:
        qid = q["query_id"]
        if qid in text_files:
            query_embeds[qid] = load_npy_vector(text_files[qid])
        else:
            missing_queries.append(qid)

    if missing_queries:
        print("以下 query 没找到 embedding，前 20 个：")
        print(missing_queries[:20])
        raise RuntimeError("query embedding 匹配失败，请检查 CLaMP3 输出文件名。")

    # 加载音频 embedding
    song_records = []
    missing_audio = []

    for song in songs:
        matched_path = None
        for key in possible_audio_keys(song):
            if key in audio_files:
                matched_path = audio_files[key]
                break

        if matched_path is None:
            missing_audio.append({
                "song_id": song.get("song_id"),
                "audio_path": song.get("audio_path"),
                "tried_keys": possible_audio_keys(song),
            })
            continue

        emb = load_npy_vector(matched_path)
        song_records.append({
            "song": song,
            "embedding": emb,
        })

    print(f"matched audio embeddings: {len(song_records)} / {len(songs)}")
    print(f"missing audio embeddings: {len(missing_audio)}")

    if len(song_records) == 0:
        print("音频 embedding 一个都没匹配上。请把 CLaMP3 输出目录结构截图/文件名发我。")
        raise RuntimeError("audio embedding 匹配失败。")

    audio_matrix = np.stack([r["embedding"] for r in song_records], axis=0)

    results = []
    csv_rows = []

    for q in queries:
        qid = q["query_id"]
        q_emb = query_embeds[qid]

        # cosine，因为两边都已经 normalize，所以直接点积
        scores = audio_matrix @ q_emb
        top_idx = np.argsort(-scores)[:TOP_K]

        top_k = []
        for rank, idx in enumerate(top_idx, start=1):
            song = song_records[int(idx)]["song"]
            score = float(scores[int(idx)])
            row = {
                "method": "clamp3",
                "query_id": qid,
                "base_query_id": q.get("base_query_id", q.get("id", "")),
                "strategy": q.get("strategy", ""),
                "language": q.get("language", ""),
                "rank": rank,
                "song_id": song.get("song_id", ""),
                "score": score,
                "music_v": song.get("music_va", ["", ""])[0],
                "music_a": song.get("music_va", ["", ""])[1],
                "display_tags": "|".join(song.get("display_tags", [])),
                "audio_path": song.get("audio_path", ""),
            }
            csv_rows.append(row)

            top_k.append({
                "rank": rank,
                "song_id": song.get("song_id", ""),
                "score": score,
                "music_va": song.get("music_va", []),
                "display_tags": song.get("display_tags", []),
                "audio_path": song.get("audio_path", ""),
            })

        results.append({
            "method": "clamp3",
            "query_id": qid,
            "base_query_id": q.get("base_query_id", q.get("id", "")),
            "strategy": q.get("strategy", ""),
            "language": q.get("language", ""),
            "prompt_en": q.get("prompt_en", ""),
            "top_k": top_k,
        })

    json_path = OUTPUT_DIR / "clamp3_top5.json"
    csv_path = OUTPUT_DIR / "clamp3_top5.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "method", "query_id", "base_query_id", "strategy", "language",
            "rank", "song_id", "score", "music_v", "music_a", "display_tags", "audio_path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("CLaMP3 ranking done.")
    print(f"- {json_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()
