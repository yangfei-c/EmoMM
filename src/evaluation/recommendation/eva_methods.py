from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# 路径
cur_dir = Path(__file__).resolve().parent
src_dir = cur_dir.parent.parent
project_root = src_dir.parent
QUERY_JSON = cur_dir / "query_test_32.json"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from application.web.service import recommend, build_tag_prior
from application.config import strategies,strength

GAMMA = 1.0
TOP_K = 5
diversity_mode = "关闭"


def load_queries():
    with open(QUERY_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def va_dist_at_k(items, mu_q, k: int) -> float:
    selected = items[:k]
    mu = np.asarray(mu_q, dtype=np.float32)

    dists = []
    for item in selected:
        z = np.asarray(item["va"], dtype=np.float32)
        dists.append(float(np.sum((z - mu) ** 2)))
    return float(np.mean(dists))


def tag_align_at_k(items, tau, k: int) -> float:
    selected = items[:k]
    tau = np.asarray(tau, dtype=np.float32)

    scores = []
    for item in selected:
        y = np.asarray(item["tag26"], dtype=np.float32)
        scores.append(float(y @ tau))

    return float(np.mean(scores))


def top1_info(items):
    x = items[0]
    return (
        x.get("song_id", ""),
        str(x.get("display_tags", "")),
        str([round(float(value), 2) for value in x.get("va", [])]),
    )


def run_one_condition(query, strategy):
    text = query["text"]

    state_uniform, recs_uniform, _, _ = recommend(
        text=text,
        strategy=strategy,
        strength=strength,
        gamma=GAMMA,
        top_k=TOP_K,
        diversity_mode=diversity_mode,
        prior_mode="uniform",
    )

    state_prototype, recs_prototype, _, _ = recommend(
        text=text,
        strategy=strategy,
        strength=strength,
        gamma=GAMMA,
        top_k=TOP_K,
        diversity_mode=diversity_mode,
        prior_mode="prototype",
    )

    tau_eval_uniform, labels_uniform = build_tag_prior(
        strategy=strategy,
        mu_q=state_uniform["mu_q"],
        prior_mode="uniform",
    )

    tau_eval_proto, labels_prototype = build_tag_prior(
        strategy=strategy,
        mu_q=state_uniform["mu_q"],
        prior_mode="prototype",
    )

    methods = {
        "VA-only": recs_uniform["VA-only"],
        "Tag-only": recs_uniform["Tag-only"],
        "VA+Tag-Uniform": recs_uniform["VA + Tag"],
        "VA+Tag+Prototype": recs_prototype["VA + Tag"],
    }

    rows = []

    for method, items in methods.items():
        top1_song, top1_tags, top1_va = top1_info(items)

        rows.append({
            "query_id": f"{query['id']}_{strategy}",
            "id": query["id"],
            "language": query["language"],
            "quadrant": query["quadrant"],
            "quadrant_name": query["quadrant_name"],
            "expected_va": query["expected_va"],
            "sample_id": query["sample_id"],
            "text": text,
            "strategy": strategy,
            "method": method,

            "z_q": str([round(float(v), 4) for v in state_uniform["z_q"]]),
            "mu_q": str([round(float(v), 4) for v in state_uniform["mu_q"]]),

            "uniform_labels": ", ".join(labels_uniform),
            "prototype_labels": ", ".join(labels_prototype),

            "VA-Dist@1": va_dist_at_k(items, state_uniform["mu_q"], 1),
            "VA-Dist@5": va_dist_at_k(items, state_uniform["mu_q"], 5),

            "Tag-Align-Uniform@1": tag_align_at_k(items, tau_eval_uniform, 1),
            "Tag-Align-Uniform@5": tag_align_at_k(items, tau_eval_uniform, 5),

            "Tag-Align-Prototype@1": tag_align_at_k(items, tau_eval_proto, 1),
            "Tag-Align-Prototype@5": tag_align_at_k(items, tau_eval_proto, 5),

            "top1_song": top1_song,
            "top1_tags": top1_tags,
            "top1_va": top1_va,
        })

    return rows


def main():
    queries = load_queries()
    all_rows = []

    for query in queries:
        for strategy in strategies:
            print(f"Running: {query['id']} | {strategy}")
            all_rows.extend(run_one_condition(query, strategy))

    df = pd.DataFrame(all_rows)

    out_dir = (
        project_root
        / "outputs"
        / "results"
        / "recommendation_eval"
        / "sanity"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / "8query_detail.csv"
    summary_path = out_dir / "8query_summary.csv"
    paper_table_path = out_dir / "8query_main_table.csv"

    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby("method")
        .agg({
            "VA-Dist@1": "mean",
            "VA-Dist@5": "mean",
            "Tag-Align-Uniform@1": "mean",
            "Tag-Align-Uniform@5": "mean",
            "Tag-Align-Prototype@1": "mean",
            "Tag-Align-Prototype@5": "mean",
        })
        .reset_index()
    )

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    paper = summary[[
        "method",
        "VA-Dist@1",
        "VA-Dist@5",
        "Tag-Align-Prototype@1",
        "Tag-Align-Prototype@5",
    ]].copy()

    paper = paper.rename(columns={
        "method": "Method",
        "Tag-Align-Prototype@1": "Tag-Align@1 ↑",
        "Tag-Align-Prototype@5": "Tag-Align@5 ↑",
        "VA-Dist@1": "VA-Dist@1 ↓",
        "VA-Dist@5": "VA-Dist@5 ↓",
    })

    method_order = ["VA-only", "Tag-only", "VA+Tag-Uniform", "VA+Tag+Prototype"]
    paper["Method"] = pd.Categorical(paper["Method"], categories=method_order, ordered=True)
    paper = paper.sort_values("Method")

    paper.to_csv(paper_table_path, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(detail_path)
    print(summary_path)
    print(paper_table_path)

    print("\nPaper main table:")
    print(paper)


if __name__ == "__main__":
    main()