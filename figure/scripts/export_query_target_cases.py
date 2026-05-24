from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
QUERIES_JSON = ROOT / "eval" / "queries_zh16.json"
OURS_CSV = ROOT / "eval" / "ours" / "ours_top5.csv"
OUTPUT_DIR = ROOT / "figure" / "data" / "04_query_target_cases"
OUTPUT_CSV = OUTPUT_DIR / "query_cases.csv"
OUTPUT_MD = OUTPUT_DIR / "query_cases.md"

CSV_HEADER = [
    "query_id",
    "strategy",
    "strategy_zh",
    "current_state",
    "regulation_goal",
    "text",
    "strength",
    "strategy_tags",
    "zq_valence",
    "zq_arousal",
    "mu_valence",
    "mu_arousal",
    "top1_song_id",
    "top1_score",
    "top1_valence",
    "top1_arousal",
    "top1_display_tags",
    "top5_song_ids",
    "top5_scores",
    "top5_display_tags",
]

STRATEGY_GROUPS = {
    "maintain": ["q01", "q05", "q09", "q13"],
    "calm": ["q02", "q06", "q10", "q14"],
    "comfort": ["q03", "q07", "q11", "q15"],
    "energize": ["q04", "q08", "q12", "q16"],
}

STRATEGY_ZH = {
    "maintain": "保持当前状态",
    "calm": "平静下来",
    "comfort": "获得安慰",
    "energize": "振作起来",
}


def read_queries() -> list[dict[str, Any]]:
    return json.loads(QUERIES_JSON.read_text(encoding="utf-8"))


def parse_list(value: str) -> list[float] | None:
    if not value:
        return None
    try:
        arr = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(arr, list) and len(arr) >= 2:
        return [float(arr[0]), float(arr[1])]
    return None


def read_ours_rows() -> list[dict[str, str]]:
    with OURS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("query_id")]


def group_by_query(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)
    return grouped


def to_text(query: dict[str, Any]) -> str:
    text = query.get("text")
    if text:
        return text
    return f"{query.get('current_state', '')}{query.get('regulation_goal', '')}"


def join_strategy_tags(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(x) for x in value)
    return str(value) if value is not None else ""


def fmt2(value: str) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def normalize_tags(value: str) -> str:
    return value.replace("|", ",") if value else ""


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda r: float(r.get("score", "-inf")), reverse=True)


def build_case_row(query: dict[str, Any], rows: list[dict[str, str]] | None) -> dict[str, str]:
    base = {
        "query_id": query.get("query_id", ""),
        "strategy": query.get("strategy", ""),
        "strategy_zh": query.get("strategy_zh", ""),
        "current_state": query.get("current_state", ""),
        "regulation_goal": query.get("regulation_goal", ""),
        "text": to_text(query),
        "strength": str(query.get("strength", "")),
        "strategy_tags": join_strategy_tags(query.get("strategy_tags")),
    }

    if not rows:
        base.update({
            "zq_valence": "",
            "zq_arousal": "",
            "mu_valence": "",
            "mu_arousal": "",
            "top1_song_id": "",
            "top1_score": "",
            "top1_valence": "",
            "top1_arousal": "",
            "top1_display_tags": "",
            "top5_song_ids": "",
            "top5_scores": "",
            "top5_display_tags": "",
        })
        return base

    sorted_rows = sort_rows(rows)
    top1 = sorted_rows[0]
    zq = parse_list(top1.get("z_q", "")) or ["", ""]
    mu = parse_list(top1.get("mu_q", "")) or ["", ""]
    top1_va = parse_list(top1.get("va", "")) or ["", ""]

    top5 = sorted_rows[:5]
    top5_song_ids = ",".join(r.get("song_id", "") for r in top5)
    top5_scores = ",".join(r.get("score", "") for r in top5)
    top5_display_tags = ",".join(normalize_tags(r.get("display_tags", "")) for r in top5)

    base.update({
        "zq_valence": fmt2(zq[0]),
        "zq_arousal": fmt2(zq[1]),
        "mu_valence": fmt2(mu[0]),
        "mu_arousal": fmt2(mu[1]),
        "top1_song_id": top1.get("song_id", ""),
        "top1_score": top1.get("score", ""),
        "top1_valence": fmt2(top1_va[0]),
        "top1_arousal": fmt2(top1_va[1]),
        "top1_display_tags": normalize_tags(top1.get("display_tags", "")),
        "top5_song_ids": top5_song_ids,
        "top5_scores": top5_scores,
        "top5_display_tags": top5_display_tags,
    })
    return base


def build_md_section_overview(cases: list[dict[str, str]]) -> list[str]:
    lines = [
        "## 一、16 条查询文本概览",
        "",
        "| query_id | strategy | strategy_zh | current_state | regulation_goal | strength |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        lines.append(
            "| {query_id} | {strategy} | {strategy_zh} | {current_state} | {regulation_goal} | {strength} |".format(
                **case
            )
        )
    return lines


def build_md_section_distribution() -> list[str]:
    lines = [
        "## 二、策略分布",
        "",
        "| strategy | strategy_zh | query_id 列表 | 数量 |",
        "| --- | --- | --- | --- |",
    ]
    for strategy, ids in STRATEGY_GROUPS.items():
        ids_text = "、".join(ids)
        lines.append(
            "| {strategy} | {strategy_zh} | {ids} | {count} |".format(
                strategy=strategy,
                strategy_zh=STRATEGY_ZH.get(strategy, ""),
                ids=ids_text,
                count=len(ids),
            )
        )
    return lines


def fmt_pair(valence: str, arousal: str) -> str:
    if valence == "" or arousal == "":
        return ""
    return f"({valence}, {arousal})"


def build_md_section_examples(cases: list[dict[str, str]]) -> list[str]:
    strategy_pick = {
        "maintain": "q01",
        "calm": "q02",
        "comfort": "q03",
        "energize": "q04",
    }
    case_map = {case["query_id"]: case for case in cases}
    lines = [
        "## 三、代表性案例",
        "",
        "| 策略 | 用户文本 | z_q | μ_q | Top-1 song_id | Top-1 VA | Top-1 tags | score |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for strategy, qid in strategy_pick.items():
        case = case_map.get(qid)
        if not case:
            continue
        lines.append(
            "| {strategy} | {text} | {zq} | {mu} | {song} | {va} | {tags} | {score} |".format(
                strategy=strategy,
                text=case["text"],
                zq=fmt_pair(case["zq_valence"], case["zq_arousal"]),
                mu=fmt_pair(case["mu_valence"], case["mu_arousal"]),
                song=case["top1_song_id"],
                va=fmt_pair(case["top1_valence"], case["top1_arousal"]),
                tags=case["top1_display_tags"],
                score=case["top1_score"],
            )
        )
    return lines


def main() -> None:
    queries = read_queries()
    ours_rows = read_ours_rows()
    grouped = group_by_query(ours_rows)

    cases = [build_case_row(q, grouped.get(q.get("query_id", ""))) for q in queries]
    missing = sum(1 for case in cases if case["top1_song_id"] == "")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(cases)

    md_lines: list[str] = ["# 查询文本与目标情绪案例导出结果", ""]
    md_lines.extend(build_md_section_overview(cases))
    md_lines.append("")
    md_lines.append(f"缺失推荐结果的查询数量：{missing}")
    md_lines.append("")
    md_lines.extend(build_md_section_distribution())
    md_lines.append("")
    md_lines.extend(build_md_section_examples(cases))
    md_lines.append("")

    OUTPUT_MD.write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
