from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


EVAL_REC_DIR = Path(__file__).resolve().parent

# src
SRC_DIR = EVAL_REC_DIR.parent

# 项目根目录
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# 实验参数
STRENGTH = 0.5
GAMMA = 1.0
TOP_K = 5

EXPERIMENT_STRATEGIES = ["maintain", "calm", "energize", "comfort"]


# 输入输出路径
QUERY_JSON = EVAL_REC_DIR / "data" / "queries.json"
OUT_DIR = SRC_DIR / "eval_rec" / "data"


def load_queries():
    with open(QUERY_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(rows, filename: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"saved: {path}")


def fmt_pair(values: Sequence[float]) -> str:
    return f"[{float(values[0]):.2f}, {float(values[1]):.2f}]"


def get_quadrant(z: Sequence[float]) -> str:
    v, a = float(z[0]), float(z[1])
    if v >= 0 and a >= 0:
        return "HVHA"
    if v >= 0 and a < 0:
        return "HVLA"
    if v < 0 and a >= 0:
        return "LVHA"
    return "LVLA"