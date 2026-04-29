from __future__ import annotations

import pandas as pd

from cfg_metrics import OUT_DIR, TOP_K


def main():
    input_path = OUT_DIR / "metric_topk_results.csv"
    df = pd.read_csv(input_path)

    va_col = f"VA-Dist@{TOP_K} ↓"
    tag_col = f"Tag-Align@{TOP_K} ↑"

    # 每个 query-condition-method 的 Top-K 平均指标
    query_level = (
        df.groupby(
            [
                "Query ID",
                "Base ID",
                "Quadrant",
                "Quadrant Name",
                "Language",
                "Strategy",
                "Method",
            ],
            as_index=False,
        )
        .agg(
            **{
                va_col: ("VA distance", "mean"),
                tag_col: ("Tag score", "mean"),
            }
        )
    )

    # 总体主表
    summary = (
        query_level.groupby("Method", as_index=False)
        .agg(
            **{
                va_col: (va_col, "mean"),
                tag_col: (tag_col, "mean"),
            }
        )
    )

    # 分策略表
    by_strategy = (
        query_level.groupby(["Strategy", "Method"], as_index=False)
        .agg(
            **{
                va_col: (va_col, "mean"),
                tag_col: (tag_col, "mean"),
            }
        )
    )

    query_level_path = OUT_DIR / "metric_query_level.csv"
    summary_path = OUT_DIR / "metric_summary.csv"
    strategy_path = OUT_DIR / "metric_by_strategy.csv"

    query_level.to_csv(query_level_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    by_strategy.to_csv(strategy_path, index=False, encoding="utf-8-sig")

    print("\nsaved:")
    print(query_level_path)
    print(summary_path)
    print(strategy_path)


if __name__ == "__main__":
    main()