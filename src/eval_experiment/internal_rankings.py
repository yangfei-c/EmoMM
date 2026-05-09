"""
输入：
    query_test_32.json
    music_eval_db.json

    3 个内部方法：
      1. VA-only
      2. Tag-only
      3. Ours: VA+Tag

输出：
  src/eval_experiment/outputs/rankings/va_only_top5.json
  src/eval_experiment/outputs/rankings/tag_only_top5.json
  src/eval_experiment/outputs/rankings/ours_vatag_top5.json
  src/eval_experiment/outputs/rankings/all_internal_rankings_top5.csv
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ===== 1. 路径配置 =====
ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

QUERY_PATH = ROOT / "src" / "eval_experiment" / "data" / "query_test_32.json"
MUSIC_DB_PATH = ROOT / "src" / "eval_experiment" / "data" / "music_eval_db.json"

OUTPUT_DIR = ROOT / "src" / "eval_experiment" / "outputs" / "rankings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5

# 标签原型参数
TOP_PERCENT = 0.05
TAU_ALPHA = 10.0

# Ours 权重
LAMBDA_VA = 0.5
LAMBDA_TAG = 0.5


# ===== 2. 工具函数 =====
def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_expected_va(expected_va: str) -> Tuple[float, float]:
    """
    将 '+V,+A' 这类象限标签转换为粗略 VA 坐标。
    注意：这是 sanity ranking 用的近似值。
    正式论文实验建议后续替换成 TextVA 模型输出。
    """
    expected_va = expected_va.replace(" ", "")
    if "+V" in expected_va:
        v = 0.75
    elif "-V" in expected_va:
        v = 0.25
    else:
        v = 0.50

    if "+A" in expected_va:
        a = 0.75
    elif "-A" in expected_va:
        a = 0.25
    else:
        a = 0.50

    return v, a


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def get_current_va_from_query(q: Dict[str, Any]) -> np.ndarray:
    """
    当前版本：优先读 query 中的 current_va；否则根据 expected_current_va / expected_va 粗略映射。
    后续你可以在这里接入真实 TextVA 模型输出。
    """
    if "current_va" in q and isinstance(q["current_va"], list) and len(q["current_va"]) == 2:
        return np.array(q["current_va"], dtype=np.float32)

    expected = q.get("expected_current_va") or q.get("expected_va")
    if expected is None:
        raise ValueError(f"query 缺少 expected_va/current_va: {q.get('query_id') or q.get('id')}")

    return np.array(parse_expected_va(expected), dtype=np.float32)


def get_target_va(current_va: np.ndarray, strategy: str) -> np.ndarray:
    """
    根据策略生成目标 VA。
    这里采用简单、可解释、可复现的策略变换，方便先跑通实验。
    后续如果你已有正式策略函数，可以替换这里。
    """
    v, a = float(current_va[0]), float(current_va[1])

    if strategy == "calm":
        # 降低唤醒度，轻微提升愉悦度
        target = [clip01(v + 0.10), clip01(a - 0.35)]

    elif strategy == "excite":
        # 提高唤醒度，同时轻微提升愉悦度
        target = [clip01(v + 0.10), clip01(a + 0.35)]

    elif strategy == "comfort":
        # 保留一定同质性，轻微提升愉悦度，避免突然改变
        target = [clip01(v + 0.15), clip01(a - 0.10)]

    elif strategy == "maintain":
        # 维持当前状态
        target = [clip01(v), clip01(a)]

    else:
        raise ValueError(f"未知 strategy: {strategy}")

    return np.array(target, dtype=np.float32)


def squared_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum((a - b) ** 2, axis=-1)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def rank_norm(scores: np.ndarray) -> np.ndarray:
    """
    将分数转成 [0,1] 的排序归一化。
    分数越大越好。
    """
    order = np.argsort(-scores)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(scores), dtype=np.float32)

    if len(scores) <= 1:
        return np.ones_like(scores, dtype=np.float32)

    # rank 0 最好 -> 1.0；rank n-1 最差 -> 0.0
    return 1.0 - ranks / (len(scores) - 1)


def build_tag_prototypes(
    music_va: np.ndarray,
    tag_vectors: np.ndarray,
    top_percent: float = 0.05,
) -> np.ndarray:
    """
    为每个 tag 维度计算一个 VA 原型。
    每个标签取该标签概率最高的 top_percent 音乐，然后按标签概率加权平均 VA。
    返回 shape = [num_tags, 2]
    """
    num_songs, num_tags = tag_vectors.shape
    k = max(1, int(math.ceil(num_songs * top_percent)))

    prototypes = []
    global_mean = np.mean(music_va, axis=0)

    for j in range(num_tags):
        scores = tag_vectors[:, j]
        top_idx = np.argsort(-scores)[:k]
        weights = scores[top_idx]

        if np.sum(weights) <= 1e-8:
            proto = global_mean
        else:
            proto = np.sum(music_va[top_idx] * weights[:, None], axis=0) / (np.sum(weights) + 1e-12)

        prototypes.append(proto)

    return np.asarray(prototypes, dtype=np.float32)


def get_target_tag_distribution(target_va: np.ndarray, tag_prototypes: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    """
    tau_j = softmax(-alpha * ||target_va - c_j||^2)
    """
    d2 = squared_dist(tag_prototypes, target_va[None, :])
    return softmax(-alpha * d2).astype(np.float32)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-scores)[:k]


def format_topk_result(
    method: str,
    q: Dict[str, Any],
    current_va: np.ndarray,
    target_va: np.ndarray,
    top_idx: np.ndarray,
    scores: np.ndarray,
    songs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    query_id = q.get("query_id") or q.get("id")
    return {
        "method": method,
        "query_id": query_id,
        "base_query_id": q.get("base_query_id", q.get("id")),
        "strategy": q.get("strategy"),
        "language": q.get("language"),
        "text": q.get("text"),
        "current_va": [float(current_va[0]), float(current_va[1])],
        "target_va": [float(target_va[0]), float(target_va[1])],
        "top_k": [
            {
                "rank": rank + 1,
                "song_id": songs[int(i)]["song_id"],
                "score": float(scores[int(i)]),
                "music_va": songs[int(i)]["music_va"],
                "display_tags": songs[int(i)].get("display_tags", []),
                "audio_path": songs[int(i)].get("audio_path", ""),
            }
            for rank, i in enumerate(top_idx)
        ],
    }


# ===== 3. 主流程 =====
def main() -> None:
    queries = load_json(QUERY_PATH)
    songs = load_json(MUSIC_DB_PATH)

    if not isinstance(queries, list):
        raise ValueError("query_test_32.json 顶层应该是 list。")
    if not isinstance(songs, list):
        raise ValueError("music_eval_db.json 顶层应该是 list。")

    print(f"queries: {len(queries)}")
    print(f"songs: {len(songs)}")

    music_va = np.array([s["music_va"] for s in songs], dtype=np.float32)
    tag_vectors = np.array([s["tag_vector_26d"] for s in songs], dtype=np.float32)

    if music_va.ndim != 2 or music_va.shape[1] != 2:
        raise ValueError(f"music_va shape 异常: {music_va.shape}")
    if tag_vectors.ndim != 2 or tag_vectors.shape[1] != 26:
        raise ValueError(f"tag_vectors shape 异常: {tag_vectors.shape}")

    print("building tag prototypes...")
    tag_prototypes = build_tag_prototypes(music_va, tag_vectors, TOP_PERCENT)

    va_results = []
    tag_results = []
    ours_results = []

    rows_for_csv = []

    for q in queries:
        current_va = get_current_va_from_query(q)
        target_va = get_target_va(current_va, q["strategy"])
        tau = get_target_tag_distribution(target_va, tag_prototypes, TAU_ALPHA)

        # 1) VA-only
        va_scores = -squared_dist(music_va, target_va[None, :])
        va_top = topk_indices(va_scores, TOP_K)
        va_res = format_topk_result("va_only", q, current_va, target_va, va_top, va_scores, songs)
        va_results.append(va_res)

        # 2) Tag-only
        tag_scores = tag_vectors @ tau
        tag_top = topk_indices(tag_scores, TOP_K)
        tag_res = format_topk_result("tag_only", q, current_va, target_va, tag_top, tag_scores, songs)
        tag_results.append(tag_res)

        # 3) Ours: VA+Tag
        ours_scores = LAMBDA_VA * rank_norm(va_scores) + LAMBDA_TAG * rank_norm(tag_scores)
        ours_top = topk_indices(ours_scores, TOP_K)
        ours_res = format_topk_result("ours_vatag", q, current_va, target_va, ours_top, ours_scores, songs)
        ours_results.append(ours_res)

        for res in [va_res, tag_res, ours_res]:
            for item in res["top_k"]:
                rows_for_csv.append({
                    "method": res["method"],
                    "query_id": res["query_id"],
                    "base_query_id": res["base_query_id"],
                    "strategy": res["strategy"],
                    "language": res["language"],
                    "current_v": res["current_va"][0],
                    "current_a": res["current_va"][1],
                    "target_v": res["target_va"][0],
                    "target_a": res["target_va"][1],
                    "rank": item["rank"],
                    "song_id": item["song_id"],
                    "score": item["score"],
                    "music_v": item["music_va"][0],
                    "music_a": item["music_va"][1],
                    "display_tags": "|".join(item.get("display_tags", [])),
                    "audio_path": item.get("audio_path", ""),
                })

    save_json(OUTPUT_DIR / "va_only_top5.json", va_results)
    save_json(OUTPUT_DIR / "tag_only_top5.json", tag_results)
    save_json(OUTPUT_DIR / "ours_vatag_top5.json", ours_results)

    csv_path = OUTPUT_DIR / "all_internal_rankings_top5.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "method", "query_id", "base_query_id", "strategy", "language",
            "current_v", "current_a", "target_v", "target_a",
            "rank", "song_id", "score", "music_v", "music_a", "display_tags", "audio_path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_for_csv)

    print("done.")
    print(f"output dir: {OUTPUT_DIR}")
    print(f"- {OUTPUT_DIR / 'va_only_top5.json'}")
    print(f"- {OUTPUT_DIR / 'tag_only_top5.json'}")
    print(f"- {OUTPUT_DIR / 'ours_vatag_top5.json'}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()
