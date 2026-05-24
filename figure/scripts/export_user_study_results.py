from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PAPER_MAIN_ZH = ROOT / "eval" / "user_study" / "downloaded_responses" / "paper_tables" / "paper_main_table_zh.md"
PAPER_MAIN_STATS = ROOT / "eval" / "user_study" / "downloaded_responses" / "paper_tables" / "paper_main_table_stats.csv"
OUTPUT_DIR = ROOT / "figure" / "data" / "05_user_study_results"
OUTPUT_CSV = OUTPUT_DIR / "user_study_results.csv"
OUTPUT_MD = OUTPUT_DIR / "user_study_results.md"

CSV_HEADER = [
    "result_level",
    "strategy",
    "method",
    "n",
    "mean_score",
    "std_score",
    "median_score",
    "best_count",
    "total_count",
    "best_ratio",
    "source",
]

METHOD_ORDER = ["Ours", "CLaMP3", "CLAP"]
STRATEGY_ORDER = ["calm", "comfort", "energize", "maintain"]


def parse_overall_stats(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("method"):
                continue
            rows.append(
                {
                    "result_level": "overall",
                    "strategy": "overall",
                    "method": row["method"],
                    "n": row["n"],
                    "mean_score": row["mean"],
                    "std_score": row["std"],
                    "median_score": row["median"],
                    "best_count": row["preferred_count"],
                    "total_count": row["total_trials"],
                    "best_ratio": row["preferred_ratio"],
                    "source": str(path),
                }
            )
    return rows


def clean_cell(value: str) -> str:
    return value.replace("**", "").strip()


def parse_mean_std(text: str) -> tuple[str, str]:
    cleaned = clean_cell(text)
    if "±" in cleaned:
        mean, std = cleaned.split("±", 1)
        return mean.strip(), std.strip()
    return cleaned, ""


def parse_best_count(text: str) -> tuple[str, str]:
    cleaned = clean_cell(text)
    if "/" in cleaned:
        left, right = cleaned.split("/", 1)
        return left.strip(), right.strip()
    return cleaned, ""


def parse_ratio(text: str) -> str:
    cleaned = clean_cell(text).replace("%", "")
    return cleaned.strip()


def parse_by_strategy_md(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) != 6:
            continue
        strategy, method, n, score_text, best_text, ratio_text = parts
        if strategy not in STRATEGY_ORDER:
            continue
        if method not in METHOD_ORDER:
            continue
        mean, std = parse_mean_std(score_text)
        best_count, total_count = parse_best_count(best_text)
        rows.append(
            {
                "result_level": "by_strategy",
                "strategy": strategy,
                "method": method,
                "n": n,
                "mean_score": mean,
                "std_score": std,
                "median_score": "",
                "best_count": best_count,
                "total_count": total_count,
                "best_ratio": parse_ratio(ratio_text),
                "source": str(path),
            }
        )
    return rows


def fmt2(value: str) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return value


def fmt_ratio(value: str) -> str:
    if value == "":
        return ""
    return f"{float(value):.2f}%"


def build_md_overall(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## 一、整体用户评价结果",
        "",
        "| 方法 | N | 调节适配评分 | Median | Best Count | Best Ratio |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    row_map = {row["method"]: row for row in rows}
    for method in METHOD_ORDER:
        row = row_map.get(method)
        if not row:
            continue
        lines.append(
            "| {method} | {n} | {mean} ± {std} | {median} | {best} / {total} | {ratio} |".format(
                method=method,
                n=row["n"],
                mean=fmt2(row["mean_score"]),
                std=fmt2(row["std_score"]),
                median=fmt2(row["median_score"]),
                best=row["best_count"],
                total=row["total_count"],
                ratio=fmt_ratio(row["best_ratio"]),
            )
        )
    return lines


def build_md_by_strategy(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return [
            "## 二、分策略用户评价结果",
            "",
            "待补充真实统计。",
        ]

    lines = [
        "## 二、分策略用户评价结果",
        "",
        "| 策略 | 方法 | N | 调节适配评分 | Best Count | Best Ratio |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for strategy in STRATEGY_ORDER:
        for method in METHOD_ORDER:
            row = next((r for r in rows if r["strategy"] == strategy and r["method"] == method), None)
            if not row:
                continue
            lines.append(
                "| {strategy} | {method} | {n} | {mean} ± {std} | {best} / {total} | {ratio} |".format(
                    strategy=strategy,
                    method=method,
                    n=row["n"],
                    mean=fmt2(row["mean_score"]),
                    std=fmt2(row["std_score"]),
                    best=row["best_count"],
                    total=row["total_count"],
                    ratio=fmt_ratio(row["best_ratio"]),
                )
            )
    return lines


def build_md_explainer() -> list[str]:
    return [
        "## 三、结果解释边界",
        "- 当前结果只能说明在本离线用户评价设置下，Ours 获得最高整体平均评分和最高整体最优选择比例；",
        "- CLaMP3 结果接近 Ours，说明跨模态音乐检索模型具有竞争力；",
        "- CLAP 相对较低，可能与其未显式建模情绪调节目标有关；",
        "- 尚未进行显著性检验，因此不能写“显著优于”“全面优于”“证明优越性”。",
    ]


def build_md_usage() -> list[str]:
    return [
        "## 四、用途说明",
        "1. 论文表：整体用户评价结果；",
        "2. 论文表：分策略用户评价结果；",
        "3. 论文图6：整体评分与最优选择比例；",
        "4. 论文图7：分策略结果图；",
        "5. 专利说明书实施例中的离线用户评价结果。",
    ]


def main() -> None:
    overall_rows = parse_overall_stats(PAPER_MAIN_STATS)
    by_strategy_rows = parse_by_strategy_md(PAPER_MAIN_ZH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(overall_rows + by_strategy_rows)

    md_lines = ["# 用户评价结果导出", ""]
    md_lines.extend(build_md_overall(overall_rows))
    md_lines.append("")
    md_lines.extend(build_md_by_strategy(by_strategy_rows))
    md_lines.append("")
    md_lines.extend(build_md_explainer())
    md_lines.append("")
    md_lines.extend(build_md_usage())
    md_lines.append("")
    OUTPUT_MD.write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

