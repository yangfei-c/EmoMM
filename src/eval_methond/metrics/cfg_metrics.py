from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


# 当前文件：src/eval_methond/metrics/cfg_metrics.py
METRICS_DIR = Path(__file__).resolve().parent

# src/eval_methond
EVAL_METHOD_DIR = METRICS_DIR.parent

# src
SRC_DIR = EVAL_METHOD_DIR.parent

# 项目根目录
PROJECT_ROOT = SRC_DIR.parent
MUSIC_DB_JSON = PROJECT_ROOT / "data" / "music_db" / "semantic_db.json"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# 实验参数
STRENGTH = 0.5
GAMMA = 1.0
TOP_K = 5
RANDOM_SEED = 42

METHODS = [
    "Random",
    "VA-only",
    "Tag-only",
    "Proposed VA+Tag",
]

EXPERIMENT_STRATEGIES = [
    "maintain",
    "calm",
    "energize",
    "comfort",
]


# 输入输出路径
QUERY_JSON = EVAL_METHOD_DIR / "queries.json"
OUT_DIR = EVAL_METHOD_DIR / "result" / "metrics"


def load_queries():
    with open(QUERY_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_pair(values: Sequence[float]) -> str:
    return f"[{float(values[0]):.2f}, {float(values[1]):.2f}]"


def score_tracks(
    method: str,
    z_matrix: np.ndarray,
    y_matrix: np.ndarray,
    mu_q,
    tau,
    rng: np.random.Generator,
):
    """
    Random:
        随机推荐，作为下界。

    VA-only:
        S(i) = -||z_i - μ_q||²

    Tag-only:
        S(i) = y_i^T τ_q

    Proposed VA+Tag:
        S(i) = -||z_i - μ_q||² + γ y_i^T τ_q
    """
    mu = np.asarray(mu_q, dtype=np.float32)
    tau = np.asarray(tau, dtype=np.float32)

    va_distance = np.sum((z_matrix - mu[None, :]) ** 2, axis=1)
    va_score = -va_distance
    tag_score = y_matrix @ tau

    if method == "Random":
        score = rng.random(len(z_matrix))
    elif method == "VA-only":
        score = va_score
    elif method == "Tag-only":
        score = tag_score
    elif method == "Proposed VA+Tag":
        score = va_score + GAMMA * tag_score
    else:
        raise ValueError(f"Unknown method: {method}")

    return score, va_distance, tag_score