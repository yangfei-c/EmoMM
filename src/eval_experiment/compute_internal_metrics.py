# -*- coding: utf-8 -*-
"""
compute_internal_metrics.py

作用：
读取内部三个方法的 Top-5 推荐结果：
  src/eval_experiment/outputs/rankings/all_internal_rankings_top5.csv

并基于：
  src/eval_experiment/data/query_test_32.json
  src/eval_experiment/data/music_eval_db.json

计算内部客观指标：
  1. TargetVA-Dist@5：推荐音乐到目标 VA 的平均欧氏距离，越低越好
  2. TagAlign@5：推荐音乐标签向量与目标标签分布 tau 的平均点积，越高越好
  3. Diversity@5：Top-5 内部音乐在 28 维语义向量上的平均两两距离，越高说明越不单一

输出：
  src/eval_experiment/outputs/metrics/internal_metrics_by_query.csv
  src/eval_experiment/outputs/metrics/internal_metrics_summary.csv
  src/eval_experiment/outputs/metrics/internal_significance_tests.csv

运行：
  cd /d D:\AAAAAAAAAA_emotion\Code\EmoMusicRec
  python src\eval_experiment\compute_internal_metrics.py
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ===== 1. 路径配置 =====
ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

QUERY_PATH = ROOT / "src" / "eval_experiment" / "data" / "query_test_32.json"
MUSIC_DB_PATH = ROOT / "src" / "eval_experiment" / "data" / "music_eval_db.json"
RANKING_CSV = ROOT / "src" / "eval_experiment" / "outputs" / "rankings" / "all_internal_rankings_top5.csv"

OUTPUT_DIR = ROOT / "src" / "eval_experiment" / "outputs" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5

# 与 run_internal_rankings.py 保持一致
TOP_PERCENT = 0.05
TAU_ALPHA = 10.0


# ===== 2. 基础函数 =====
def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_expected_va(expected_va: str) -> Tuple[float, float]:
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
    if "current_va" in q and isinstance(q["current_va"], list) and len(q["current_va"]) == 2:
        return np.array(q["current_va"], dtype=np.float32)

    expected = q.get("expected_current_va") or q.get("expected_va")
    if expected is None:
        raise ValueError(f"query 缺少 expected_va/current_va: {q.get('query_id') or q.get('id')}")

    return np.array(parse_expected_va(expected), dtype=np.float32)


def get_target_va(current_va: np.ndarray, strategy: str) -> np.ndarray:
    v, a = float(current_va[0]), float(current_va[1])

    if strategy == "calm":
        target = [clip01(v + 0.10), clip01(a - 0.35)]
    elif strategy == "excite":
        target = [clip01(v + 0.10), clip01(a + 0.35)]
    elif strategy == "comfort":
        target = [clip01(v + 0.15), clip01(a - 0.10)]
    elif strategy == "maintain":
        target = [clip01(v), clip01(a)]
    else:
        raise ValueError(f"未知 strategy: {strategy}")

    return np.array(target, dtype=np.float32)


def squared_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum((a - b) ** 2, axis=-1)


def euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def build_tag_prototypes(music_va: np.ndarray, tag_vectors: np.ndarray, top_percent: float = 0.05) -> np.ndarray:
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
    d2 = squared_dist(tag_prototypes, target_va[None, :])
    return softmax(-alpha * d2).astype(np.float32)


def avg_pairwise_distance(mat: np.ndarray) -> float:
    """
    mat shape: [K, D]
    返回 K 个向量的平均两两欧氏距离。
    K<2 时返回 0。
    """
    if len(mat) < 2:
        return 0.0

    dists = []
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            dists.append(float(np.linalg.norm(mat[i] - mat[j])))

    return float(np.mean(dists)) if dists else 0.0


def holm_bonferroni(pvals: List[float]) -> List[float]:
    """
    简单 Holm-Bonferroni 校正，返回 adjusted p-values。
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = [None] * m
    prev = 0.0

    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        adj = max(adj, prev)
        adj = min(adj, 1.0)
        adjusted[idx] = adj
        prev = adj

    return adjusted


# ===== 3. 统计检验 =====
def run_significance_tests(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 metric 做：
    - Friedman：三个内部方法整体比较
    - Wilcoxon：Ours vs VA-only；Ours vs Tag-only
    """
    try:
        from scipy.stats import friedmanchisquare, wilcoxon
        scipy_ok = True
    except Exception:
        scipy_ok = False

    rows = []
    metric_names = ["target_va_dist_at_5", "tag_align_at_5", "diversity_at_5"]
    methods = ["va_only", "tag_only", "ours_vatag"]

    for metric in metric_names:
        pivot = metrics_df.pivot(index="query_id", columns="method", values=metric)

        # 只保留三种方法都有结果的 query
        pivot = pivot.dropna(subset=methods)

        if not scipy_ok:
            rows.append({
                "metric": metric,
                "test": "not_run",
                "comparison": "scipy_not_available",
                "statistic": "",
                "p_value": "",
                "p_value_holm": "",
                "n": len(pivot),
            })
            continue

        # Friedman
        stat, p = friedmanchisquare(
            pivot["va_only"].values,
            pivot["tag_only"].values,
            pivot["ours_vatag"].values,
        )
        rows.append({
            "metric": metric,
            "test": "Friedman",
            "comparison": "va_only vs tag_only vs ours_vatag",
            "statistic": float(stat),
            "p_value": float(p),
            "p_value_holm": "",
            "n": len(pivot),
        })

        # Wilcoxon post-hoc: Ours vs baselines
        posthoc = []
        for baseline in ["va_only", "tag_only"]:
            try:
                stat_w, p_w = wilcoxon(
                    pivot["ours_vatag"].values,
                    pivot[baseline].values,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                )
                posthoc.append((baseline, float(stat_w), float(p_w)))
            except Exception as e:
                posthoc.append((baseline, "", float("nan")))

        corrected = holm_bonferroni([x[2] for x in posthoc])

        for (baseline, stat_w, p_w), p_adj in zip(posthoc, corrected):
            rows.append({
                "metric": metric,
                "test": "Wilcoxon",
                "comparison": f"ours_vatag vs {baseline}",
                "statistic": stat_w,
                "p_value": p_w,
                "p_value_holm": p_adj,
                "n": len(pivot),
            })

    return pd.DataFrame(rows)


# ===== 4. 主流程 =====
def main() -> None:
    queries = load_json(QUERY_PATH)
    songs = load_json(MUSIC_DB_PATH)

    if not RANKING_CSV.exists():
        raise FileNotFoundError(f"找不到推荐结果 CSV: {RANKING_CSV}")

    rankings = pd.read_csv(RANKING_CSV)

    print(f"queries: {len(queries)}")
    print(f"songs: {len(songs)}")
    print(f"ranking rows: {len(rankings)}")

    song_map = {s["song_id"]: s for s in songs}
    query_map = {(q.get("query_id") or q.get("id")): q for q in queries}

    music_va_all = np.array([s["music_va"] for s in songs], dtype=np.float32)
    tag_vectors_all = np.array([s["tag_vector_26d"] for s in songs], dtype=np.float32)

    print("building tag prototypes...")
    tag_prototypes = build_tag_prototypes(music_va_all, tag_vectors_all, TOP_PERCENT)

    rows = []

    for (method, query_id), group in rankings.groupby(["method", "query_id"]):
        q = query_map.get(query_id)
        if q is None:
            raise ValueError(f"ranking 中的 query_id 在 query 文件里找不到: {query_id}")

        current_va = get_current_va_from_query(q)
        target_va = get_target_va(current_va, q["strategy"])
        tau = get_target_tag_distribution(target_va, tag_prototypes, TAU_ALPHA)

        # 保证按 rank 排序，只取 Top-K
        group = group.sort_values("rank").head(TOP_K)
        top_song_ids = group["song_id"].tolist()

        top_music_va = []
        top_tag_vecs = []
        top_semantic_vecs = []

        for sid in top_song_ids:
            s = song_map.get(sid)
            if s is None:
                raise ValueError(f"song_id 在 music_eval_db 中找不到: {sid}")

            top_music_va.append(s["music_va"])
            top_tag_vecs.append(s["tag_vector_26d"])

            if "semantic_vector_28d" in s:
                top_semantic_vecs.append(s["semantic_vector_28d"])
            else:
                top_semantic_vecs.append(s["music_va"] + s["tag_vector_26d"])

        top_music_va = np.array(top_music_va, dtype=np.float32)
        top_tag_vecs = np.array(top_tag_vecs, dtype=np.float32)
        top_semantic_vecs = np.array(top_semantic_vecs, dtype=np.float32)

        target_va_dist = float(np.mean(euclidean(top_music_va, target_va[None, :])))
        tag_align = float(np.mean(top_tag_vecs @ tau))
        diversity = avg_pairwise_distance(top_semantic_vecs)

        rows.append({
            "method": method,
            "query_id": query_id,
            "base_query_id": q.get("base_query_id", q.get("id")),
            "language": q.get("language"),
            "strategy": q.get("strategy"),
            "current_v": float(current_va[0]),
            "current_a": float(current_va[1]),
            "target_v": float(target_va[0]),
            "target_a": float(target_va[1]),
            "target_va_dist_at_5": target_va_dist,
            "tag_align_at_5": tag_align,
            "diversity_at_5": diversity,
            "top_song_ids": "|".join(top_song_ids),
        })

    metrics_df = pd.DataFrame(rows)

    by_query_path = OUTPUT_DIR / "internal_metrics_by_query.csv"
    metrics_df.to_csv(by_query_path, index=False, encoding="utf-8-sig")

    summary = (
        metrics_df
        .groupby("method")[["target_va_dist_at_5", "tag_align_at_5", "diversity_at_5"]]
        .agg(["mean", "std", "median"])
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    summary_path = OUTPUT_DIR / "internal_metrics_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    tests_df = run_significance_tests(metrics_df)
    tests_path = OUTPUT_DIR / "internal_significance_tests.csv"
    tests_df.to_csv(tests_path, index=False, encoding="utf-8-sig")

    print("done.")
    print(f"- {by_query_path}")
    print(f"- {summary_path}")
    print(f"- {tests_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
