# -*- coding: utf-8 -*-

import csv
import json
import sys
from pathlib import Path

import numpy as np


root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir / "src"))

from core.music_repository import load_tracks
from core.tag_prototype import compute_tag_prototypes
from core.utils import select_nearest_tags, make_prototype_prior
from core.scoring import compute_scores, topk_indices
from core.user_state import UserStateEstimator
from core.config import strategy_tag


query_json = root_dir / "eval" / "queries_zh16.json"
out_csv = root_dir / "eval" / "ours" / "ours_top5.csv"

TOP_K = 5

TAG26 = [
    "ambient", "classical", "electronic", "folk", "hiphop", "jazz",
    "metal", "pop", "rock", "soundtrack",
    "bass", "drums", "guitar", "piano", "strings", "synthesizer", "voice",
    "dark", "dream", "energetic", "epic", "film", "happy", "love", "relaxing", "sad",
]


def get_query_id(q):
    qid = q.get("query_id") or q.get("id")
    if not qid:
        raise ValueError(f"query 缺少 query_id/id: {q}")
    return str(qid)


def get_current_state_text(q):
    """
    Ours 只用 current_state 估计当前情绪 z_q。
    不能用完整 text，否则 regulation_goal 会污染当前情绪估计。
    """
    text = str(q.get("current_state") or "").strip()

    if not text:
        # 兼容旧 JSON；但新版正式实验应该有 current_state
        text = str(q.get("text") or "").strip()

    if not text:
        raise ValueError(f"{get_query_id(q)} 缺少 current_state/text")

    return text


def get_display_text(q):
    """
    页面和 CSV 记录用完整 text。
    CLAP / CLaMP3 用这个完整 text。
    """
    text = str(q.get("text") or "").strip()

    if text:
        return text

    current_state = str(q.get("current_state") or "").strip()
    regulation_goal = str(q.get("regulation_goal") or "").strip()

    return f"{current_state}{regulation_goal}".strip()


def get_strategy(q):
    strategy = str(q.get("strategy") or "").strip()

    if not strategy:
        raise ValueError(f"{get_query_id(q)} 缺少 strategy")

    return strategy


def get_strength(q):
    return float(q.get("strength", 0.5))


def get_strategy_labels(q, strategy):
    """
    标签先读 query 里的 strategy_tags。
    如果新版 JSON 没写，则用 core.config.strategy_tag。
    """
    labels = q.get("strategy_tags", None)

    if labels is None:
        labels = strategy_tag.get(strategy, [])

    if not isinstance(labels, list):
        raise ValueError(f"{get_query_id(q)} strategy_tags 必须是 list: {labels}")

    return labels


def main():
    queries = json.loads(query_json.read_text(encoding="utf-8"))

    tracks = load_tracks()
    tracks = [
        t for t in tracks
        if t.get("va") is not None and t.get("tag26") is not None
    ]

    print(f"全量音乐数量: {len(tracks)}")
    print(f"query 数量: {len(queries)}")

    z_matrix = np.asarray([t["va"] for t in tracks], dtype=np.float32)
    y_matrix = np.asarray([t["tag26"] for t in tracks], dtype=np.float32)

    prototypes = compute_tag_prototypes(tracks)

    estimator = UserStateEstimator()

    rows = []

    for qi, q in enumerate(queries, start=1):
        qid = get_query_id(q)
        strategy = get_strategy(q)
        strength = get_strength(q)

        current_state = get_current_state_text(q)
        display_text = get_display_text(q)
        regulation_goal = str(q.get("regulation_goal") or "").strip()
        strategy_zh = str(q.get("strategy_zh") or strategy).strip()

        print("=" * 80)
        print(f"[{qi}/{len(queries)}] {qid}")
        print(f"strategy: {strategy}, strength: {strength}")
        print(f"current_state: {current_state}")

        # Ours: current_state -> z_q, strategy + strength -> mu_q
        state = estimator.process(
            text=current_state,
            strategy=strategy,
            strength=strength,
        )

        z_q = state["z_q"]
        mu_q = state["mu_q"]

        labels = get_strategy_labels(q, strategy)

        if labels == ["auto"]:
            labels = select_nearest_tags(mu_q, prototypes, TAG26)

        tau = make_prototype_prior(
            mu_q=mu_q,
            labels=labels,
            tag26=TAG26,
            prototypes=prototypes,
            alpha=10.0,
        )

        scores = compute_scores(
            z_matrix=z_matrix,
            y_matrix=y_matrix,
            z_q=z_q,
            mu_q=mu_q,
            tau=tau,
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
        )

        top_idx = topk_indices(scores["total_score"], TOP_K)

        for rank, idx in enumerate(top_idx, start=1):
            idx = int(idx)
            track = tracks[idx]

            rows.append({
                "method": "ours",
                "query_id": qid,
                "rank": rank,
                "song_id": str(track["song_id"]),
                "score": round(float(scores["total_score"][idx]), 6),

                "text": display_text,
                "current_state": current_state,
                "regulation_goal": regulation_goal,
                "strategy": strategy,
                "strategy_zh": strategy_zh,
                "strength": strength,
                "z_q": json.dumps(z_q, ensure_ascii=False),
                "mu_q": json.dumps(mu_q, ensure_ascii=False),

                "display_tags": "|".join(map(str, track.get("display_tags", []))),
                "va": json.dumps(track.get("va", []), ensure_ascii=False),
                "audio_path": str(track.get("song_path") or track.get("audio_path") or ""),
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)

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
        "z_q",
        "mu_q",
        "display_tags",
        "va",
        "audio_path",
    ]

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("Ours full-library Top-5 生成完成")
    print(f"输出: {out_csv}")
    print(f"输出行数: {len(rows)}")

    expected_rows = len(queries) * TOP_K
    if len(rows) != expected_rows:
        print(f"Warning: 期望 {expected_rows} 行，实际 {len(rows)} 行")

    empty_audio_path = sum(1 for r in rows if not r["audio_path"])
    print(f"audio_path 为空数量: {empty_audio_path}")


if __name__ == "__main__":
    main()