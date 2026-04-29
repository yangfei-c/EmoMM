from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


# EmoMusicRec/src/evaluation/recommendation
cur_dir = Path(__file__).resolve().parent
# EmoMusicRec/src/evaluation
evaluation_dir = cur_dir.parent
# EmoMusicRec/src
src_dir = evaluation_dir.parent
#EmoMusicRec
project_root = src_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


strength = 0.5
gamma = 1.0
top_k = 5
random_seed = 42
strategies = ["maintain","calm","energize","comfort",]

methods = ["VA-only","Tag-only","Proposed VA+Tag",]


query_json = cur_dir / "queries.json"
music_db_json = project_root / "data" / "music_db" / "semantic_db.json"
rec_out_dir = project_root / "outputs" / "results" / "recommendation_eval" / "rec"
metric_out_dir = project_root / "outputs" / "results" / "recommendation_eval" / "metrics"


def load_queries():
    with open(query_json, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"saved: {path}")

def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
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
