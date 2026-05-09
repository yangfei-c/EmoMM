# -*- coding: utf-8 -*-

import csv
import json
import pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# 注意：这里用你当前项目里的文件名
QUERY_JSON = ROOT / "data" / "query" / "queries_zh16.json"

# CLaMP3 embedding 放在项目外，避免 PyCharm 索引卡顿
CLAMP3_EVAL_DIR = Path(r"D:\AAAA_audio_dataset\eval")

MUSIC_EMB_DIR = CLAMP3_EVAL_DIR / "mert"
QUERY_EMB_DIR = CLAMP3_EVAL_DIR / "query"

OUT_CSV = ROOT / "data" / "eval" / "ours" / "clamp3_top5.csv"

TOP_K = 5


def read_ids(folder):
    ids_json = folder / "ids.json"
    ids_txt = folder / "ids.txt"

    if ids_json.exists():
        data = json.loads(ids_json.read_text(encoding="utf-8"))
        return [str(x) for x in data]

    if ids_txt.exists():
        return [
            line.strip()
            for line in ids_txt.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return None


def load_emb_dir(folder):
    """
    递归读取 CLaMP3 embedding 目录。
    支持：
    1. *.npz: embeddings + ids
    2. *.pkl: embeddings + ids
    3. embeddings.npy + ids.json / ids.txt
    4. 多个单向量 .npy，包括子目录里的 .npy
    """
    if not folder.exists():
        raise FileNotFoundError(f"找不到 embedding 目录：{folder}")

    # 1. npz
    for p in folder.rglob("*.npz"):
        data = np.load(p, allow_pickle=True)

        emb_key = None
        for k in ["embeddings", "embedding", "features", "vectors"]:
            if k in data.files:
                emb_key = k
                break

        id_key = None
        for k in ["ids", "song_ids", "track_ids", "query_ids"]:
            if k in data.files:
                id_key = k
                break

        if emb_key and id_key:
            ids = [str(x) for x in data[id_key].tolist()]
            emb = np.asarray(data[emb_key], dtype=np.float32)
            return ids, emb

    # 2. pkl
    for p in folder.rglob("*.pkl"):
        with p.open("rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            continue

        emb_key = None
        for k in ["embeddings", "embedding", "features", "vectors"]:
            if k in data:
                emb_key = k
                break

        id_key = None
        for k in ["ids", "song_ids", "track_ids", "query_ids"]:
            if k in data:
                id_key = k
                break

        if emb_key and id_key:
            ids = [str(x) for x in data[id_key]]
            emb = np.asarray(data[emb_key], dtype=np.float32)
            return ids, emb

    # 3. ids + matrix npy
    ids = read_ids(folder)

    for p in folder.rglob("*.npy"):
        try:
            arr = np.load(p, allow_pickle=True)
        except Exception:
            continue

        if arr.ndim == 2 and ids is not None:
            if len(ids) != arr.shape[0]:
                raise ValueError(
                    f"ids 数量和 embedding 行数不一致：{len(ids)} vs {arr.shape[0]}"
                )
            return ids, np.asarray(arr, dtype=np.float32)

    # 4. 递归读取多个单向量 npy
    vec_files = sorted(folder.rglob("*.npy"))

    ids = []
    vecs = []
    bad = []

    for p in vec_files:
        try:
            arr = np.load(p, allow_pickle=True).squeeze()
        except Exception as e:
            bad.append((p, repr(e)))
            continue

        if arr.ndim != 1:
            bad.append((p, f"shape={arr.shape}, squeeze 后不是 1D"))
            continue

        # 例如 D:\...\mert\87\387.npy -> song_id = 387
        ids.append(p.stem)
        vecs.append(arr.astype(np.float32))

    if vecs:
        print(f"读取 embedding 文件数：{len(vecs)}")
        if bad:
            print(f"跳过坏文件数：{len(bad)}")
            for p, e in bad[:10]:
                print(f"BAD: {p} | {e}")

        emb = np.stack(vecs).astype(np.float32)
        return ids, emb

    raise RuntimeError(
        f"无法读取 embedding：{folder}\n"
        f"该目录下 npy 数量：{len(vec_files)}\n"
        "请确认这里是不是 CLaMP3 使用 --get_global 生成的 .npy embedding 目录。"
    )
def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    if x.ndim != 2:
        raise ValueError(f"embedding 必须是 2 维矩阵，当前 shape={x.shape}")

    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-8)


def main():
    if not QUERY_JSON.exists():
        raise FileNotFoundError(f"找不到 query 文件：{QUERY_JSON}")

    if not MUSIC_EMB_DIR.exists():
        raise FileNotFoundError(f"找不到音乐 embedding 目录：{MUSIC_EMB_DIR}")

    if not QUERY_EMB_DIR.exists():
        raise FileNotFoundError(f"找不到 query embedding 目录：{QUERY_EMB_DIR}")

    queries = json.loads(QUERY_JSON.read_text(encoding="utf-8"))

    q_ids, q_emb = load_emb_dir(QUERY_EMB_DIR)
    m_ids, m_emb = load_emb_dir(MUSIC_EMB_DIR)

    q_emb = l2_normalize(q_emb)
    m_emb = l2_normalize(m_emb)

    q_index = {qid: i for i, qid in enumerate(q_ids)}
    q_info = {q["query_id"]: q for q in queries}

    rows = []

    for q in queries:
        qid = q["query_id"]

        if qid not in q_index:
            raise KeyError(
                f"query embedding 里找不到 {qid}。\n"
                f"请检查 {QUERY_EMB_DIR} 里的 .npy 文件名是否等于 query_id。"
            )

        scores = m_emb @ q_emb[q_index[qid]]
        top_idx = np.argsort(-scores)[:TOP_K]

        for rank, idx in enumerate(top_idx, start=1):
            rows.append({
                "method": "clamp3",
                "query_id": qid,
                "rank": rank,
                "song_id": m_ids[int(idx)],
                "score": round(float(scores[int(idx)]), 6),
                "text": q_info[qid]["text"],
                "strategy": q_info[qid]["strategy"],
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("CLaMP3 Top-5 生成完成")
    print(f"输入 query: {QUERY_JSON}")
    print(f"音乐 embedding: {MUSIC_EMB_DIR}")
    print(f"文本 embedding: {QUERY_EMB_DIR}")
    print(f"输出文件: {OUT_CSV}")
    print(f"query 数量: {len(queries)}")
    print(f"输出行数: {len(rows)}")


if __name__ == "__main__":
    main()