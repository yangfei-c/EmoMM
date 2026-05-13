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

# 如果 True，会检查 query embedding 文件修改时间。
# 新版 query 后如果没重新生成 query_embd，会直接报错。
STRICT_QUERY_EMBD_TIME_CHECK = True


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
    qid = q.get("query_id") or q.get("id")
    if not qid:
        raise ValueError(f"query 缺少 query_id/id: {q}")
    return str(qid)


def get_retrieval_text(q):
    """
    CLaMP3 baseline 使用完整查询文本：
    current_state + regulation_goal。
    """
    text = str(q.get("text") or "").strip()

    if text:
        return text

    current_state = str(q.get("current_state") or "").strip()
    regulation_goal = str(q.get("regulation_goal") or "").strip()
    text = f"{current_state}{regulation_goal}".strip()

    if not text:
        raise ValueError(f"{get_query_id(q)} 缺少 text/current_state/regulation_goal")

    return text


def get_current_state(q):
    return str(q.get("current_state") or "").strip()


def get_regulation_goal(q):
    return str(q.get("regulation_goal") or "").strip()


def get_strategy_zh(q):
    return str(q.get("strategy_zh") or q.get("strategy") or "").strip()


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


def validate_query_embeddings_are_fresh(queries):
    """
    CLaMP3 top5 脚本只能读取 query_embd。
    如果 query JSON 改了但 query_embd 没重新生成，会得到旧结果。
    所以这里做两类检查：
    1. 每个 query_id 都有对应 .npy
    2. query_embd 文件时间不早于 queries_zh16.json
    """
    query_ids = [get_query_id(q) for q in queries]
    missing = []

    for qid in query_ids:
        p = QUERY_EMBD / f"{qid}.npy"
        if not p.exists():
            missing.append(str(p))

    if missing:
        print("缺少 CLaMP3 query embedding:")
        for x in missing[:20]:
            print("  ", x)
        raise FileNotFoundError(
            "CLaMP3 query_embd 不完整。请先用新版 queries_zh16.json 重新生成 query_embd。"
        )

    if not STRICT_QUERY_EMBD_TIME_CHECK:
        return

    query_mtime = QUERY_JSON.stat().st_mtime
    stale = []

    for qid in query_ids:
        p = QUERY_EMBD / f"{qid}.npy"
        if p.stat().st_mtime < query_mtime:
            stale.append(str(p))

    if stale:
        print("以下 CLaMP3 query embedding 比 queries_zh16.json 更旧，可能还是旧 query 的向量:")
        for x in stale[:20]:
            print("  ", x)

        raise RuntimeError(
            "检测到 CLaMP3 query_embd 可能过期。"
            "请删除或覆盖 D:\\AAAA_audio_dataset\\eval\\clamp3\\query_embd 下旧的 q*.npy，"
            "并用新版 queries_zh16.json 重新生成 query embedding。"
        )


def main():
    queries = json.loads(QUERY_JSON.read_text(encoding="utf-8"))

    validate_query_embeddings_are_fresh(queries)

    track_info = build_track_info()

    query_ids, query_emb = load_embd(QUERY_EMBD)
    song_ids, audio_emb = load_embd(AUDIO_EMBD)

    expected_query_ids = [get_query_id(q) for q in queries]
    query_index = {qid: i for i, qid in enumerate(query_ids)}

    print(f"query embedding: {query_emb.shape}")
    print(f"audio embedding: {audio_emb.shape}")
    print(f"audio ids 数量: {len(song_ids)}")
    print(f"query ids 数量: {len(query_ids)}")

    if query_emb.shape[1] != audio_emb.shape[1]:
        raise ValueError(f"维度不一致: query={query_emb.shape}, audio={audio_emb.shape}")

    for qid in expected_query_ids:
        if qid not in query_index:
            raise KeyError(f"query_embd 里找不到: {qid}")

    missing_meta = [sid for sid in song_ids if sid not in track_info]
    print(f"audio embedding 中找不到 metadata 的数量: {len(missing_meta)}")

    if missing_meta:
        print("示例 missing metadata song_id:")
        for sid in missing_meta[:10]:
            print("  ", sid)

    rows = []

    for q in queries:
        qid = get_query_id(q)

        scores = audio_emb @ query_emb[query_index[qid]]
        top_idx = np.argsort(-scores)[:TOP_K]

        query_text = get_retrieval_text(q)

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

                "text": query_text,
                "current_state": get_current_state(q),
                "regulation_goal": get_regulation_goal(q),
                "strategy": q.get("strategy", ""),
                "strategy_zh": get_strategy_zh(q),
                "strength": q.get("strength", 0.5),

                "display_tags": "|".join(map(str, info.get("display_tags", []))),
                "va": json.dumps(info.get("va", []), ensure_ascii=False),
                "audio_path": str(info.get("song_path", "")),
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "query_id",
        "rank",
        "song_id",
        "score",
        "text",
        "current_state",
        "regulation_goal",
        "strategy",
        "strategy_zh",
        "strength",
        "display_tags",
        "va",
        "audio_path",
    ]

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
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