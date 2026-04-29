from __future__ import annotations

import pandas as pd

from common import metric_out_dir, top_k, save_df


def main():
    input_path = metric_out_dir / "metric_topk_results.csv"
    df = pd.read_csv(input_path)

    va_col = f"VA-Dist@{top_k} ↓"
    tag_col = f"Tag-Align@{top_k} ↑"

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

    query_level_path = metric_out_dir / "metric_query_level.csv"
    summary_path = metric_out_dir / "metric_summary.csv"
    strategy_path = metric_out_dir / "metric_by_strategy.csv"
    save_df(query_level, query_level_path)
    save_df(summary, summary_path)
    save_df(by_strategy, strategy_path)
    print("\nsaved:")
    print(query_level_path)
    print(summary_path)
    print(strategy_path)


if __name__ == "__main__":
    main()