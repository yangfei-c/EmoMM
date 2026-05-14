# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd


METHOD_ORDER = ["ours", "clamp3", "clap"]

METHOD_DISPLAY_ZH = {
    "ours": "Ours",
    "clamp3": "CLaMP3",
    "clap": "CLAP",
}

METHOD_DISPLAY_EN = {
    "ours": "Ours",
    "clamp3": "CLaMP3",
    "clap": "CLAP",
}


def normalize_method_name(x):
    x = str(x or "").strip().lower()

    if x in ["ours", "our", "proposed", "proposed method"]:
        return "ours"

    if x in ["clamp3", "clamp", "cp3"]:
        return "clamp3"

    if x in ["clap", "clip"]:
        return "clap"

    return x


def read_csv_safely(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    if "method_name" not in df.columns:
        raise ValueError("CSV 缺少 method_name 列，无法还原真实方法。")

    if "token" not in df.columns:
        raise ValueError("CSV 缺少 token 列。")

    if "trial_index" not in df.columns:
        raise ValueError("CSV 缺少 trial_index 列。")

    df["method_key"] = df["method_name"].apply(normalize_method_name)

    return df


def detect_score_mode(df):
    if "regulation_suitability" in df.columns:
        return "single"

    old_fields = ["regulation_fit", "helpfulness", "satisfaction"]
    if all(x in df.columns for x in old_fields):
        return "three_dim"

    raise ValueError(
        "无法识别评分字段。需要 regulation_suitability，"
        "或 regulation_fit/helpfulness/satisfaction 三列。"
    )


def prepare_scores(df, mode):
    df = df.copy()

    if mode == "single":
        df["main_score"] = pd.to_numeric(df["regulation_suitability"], errors="coerce")
        df = df.dropna(subset=["main_score"])

    else:
        for col in ["regulation_fit", "helpfulness", "satisfaction"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["main_score"] = df[["regulation_fit", "helpfulness", "satisfaction"]].mean(axis=1)
        df = df.dropna(subset=["main_score"])

    return df


def prepare_preference(df):
    df = df.copy()

    if "is_preferred" in df.columns:
        df["preferred"] = pd.to_numeric(df["is_preferred"], errors="coerce").fillna(0).astype(int)
        return df

    if "preference" in df.columns and "method_label" in df.columns:
        df["preferred"] = (df["preference"].astype(str) == df["method_label"].astype(str)).astype(int)
        return df

    raise ValueError("CSV 缺少 is_preferred，且无法用 preference + method_label 计算偏好。")


def participant_trial_count(df):
    participant_count = df["token"].nunique()
    trial_count = df[["token", "trial_index"]].drop_duplicates().shape[0]
    return participant_count, trial_count


def aggregate_main_table(df, mode):
    rows = []

    for method in METHOD_ORDER:
        sub = df[df["method_key"] == method]

        if sub.empty:
            continue

        score_n = int(sub["main_score"].count())
        mean = float(sub["main_score"].mean())
        std = float(sub["main_score"].std(ddof=1)) if score_n > 1 else 0.0
        median = float(sub["main_score"].median())

        preferred_count = int(sub["preferred"].sum())
        total_trials = int(sub[["token", "trial_index"]].drop_duplicates().shape[0])
        preferred_ratio = preferred_count / total_trials * 100 if total_trials else 0.0

        row = {
            "method_key": method,
            "method": METHOD_DISPLAY_EN.get(method, method),
            "n": score_n,
            "mean": mean,
            "std": std,
            "median": median,
            "preferred_count": preferred_count,
            "total_trials": total_trials,
            "preferred_ratio": preferred_ratio,
        }

        if mode == "three_dim":
            for col in ["regulation_fit", "helpfulness", "satisfaction"]:
                values = sub[col].dropna()
                row[f"{col}_mean"] = float(values.mean())
                row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_strategy_table(df):
    if "strategy" not in df.columns:
        return pd.DataFrame()

    rows = []

    for strategy in sorted(df["strategy"].dropna().unique()):
        s_df = df[df["strategy"] == strategy]

        for method in METHOD_ORDER:
            sub = s_df[s_df["method_key"] == method]

            if sub.empty:
                continue

            n = int(sub["main_score"].count())
            mean = float(sub["main_score"].mean())
            std = float(sub["main_score"].std(ddof=1)) if n > 1 else 0.0
            preferred_count = int(sub["preferred"].sum())
            total_trials = int(sub[["token", "trial_index"]].drop_duplicates().shape[0])
            preferred_ratio = preferred_count / total_trials * 100 if total_trials else 0.0

            rows.append({
                "strategy": strategy,
                "method_key": method,
                "method": METHOD_DISPLAY_EN.get(method, method),
                "n": n,
                "mean": mean,
                "std": std,
                "preferred_count": preferred_count,
                "total_trials": total_trials,
                "preferred_ratio": preferred_ratio,
            })

    return pd.DataFrame(rows)


def fmt_mean_std(mean, std):
    return f"{mean:.2f} ± {std:.2f}"


def fmt_count_ratio(count, total, ratio):
    return f"{count} / {total}"


def fmt_percent(x):
    return f"{x:.2f}%"


def bold_if_best(value, best_value, text, higher_is_better=True):
    if higher_is_better and value == best_value:
        return f"**{text}**"

    if not higher_is_better and value == best_value:
        return f"**{text}**"

    return text


def make_main_table_zh(stats, mode, participant_count, trial_count):
    lines = []

    if mode == "single":
        lines.append("## 表 2 用户研究主结果")
        lines.append("")
        lines.append("| 方法 | 评分数 N | 调节适配评分 ↑ | 中位数 | 被选为最优次数 ↑ | 最优选择比例 ↑ |")
        lines.append("|---|---:|---:|---:|---:|---:|")

        best_mean = stats["mean"].max()
        best_median = stats["median"].max()
        best_pref = stats["preferred_count"].max()
        best_ratio = stats["preferred_ratio"].max()

        for _, r in stats.iterrows():
            score = bold_if_best(r["mean"], best_mean, fmt_mean_std(r["mean"], r["std"]))
            median = bold_if_best(r["median"], best_median, f"{r['median']:.2f}")
            pref = bold_if_best(r["preferred_count"], best_pref, fmt_count_ratio(r["preferred_count"], r["total_trials"], r["preferred_ratio"]))
            ratio = bold_if_best(r["preferred_ratio"], best_ratio, fmt_percent(r["preferred_ratio"]))

            lines.append(
                f"| {r['method']} | {int(r['n'])} | {score} | {median} | {pref} | {ratio} |"
            )

        lines.append("")
        lines.append(
            f"表注：表中调节适配评分为 1–7 分，数值越高表示参与者越认为该音乐适合帮助其从当前情绪调节到目标状态。"
            f"最优选择比例表示该方法在每个 trial 的 A/B/C 三首匿名音乐中被参与者选为最愿意使用音乐的比例。"
            f"当前结果基于 {participant_count} 名参与者、{trial_count} 个情绪调节场景。"
        )

    else:
        lines.append("## 表 2 用户研究主结果")
        lines.append("")
        lines.append("| 方法 | 评分数 N | 符合性 ↑ | 帮助性 ↑ | 满意度 ↑ | 综合评分 ↑ | 被选为最优次数 ↑ | 最优选择比例 ↑ |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

        best_fit = stats["regulation_fit_mean"].max()
        best_help = stats["helpfulness_mean"].max()
        best_sat = stats["satisfaction_mean"].max()
        best_main = stats["mean"].max()
        best_pref = stats["preferred_count"].max()
        best_ratio = stats["preferred_ratio"].max()

        for _, r in stats.iterrows():
            fit = bold_if_best(r["regulation_fit_mean"], best_fit, fmt_mean_std(r["regulation_fit_mean"], r["regulation_fit_std"]))
            help_ = bold_if_best(r["helpfulness_mean"], best_help, fmt_mean_std(r["helpfulness_mean"], r["helpfulness_std"]))
            sat = bold_if_best(r["satisfaction_mean"], best_sat, fmt_mean_std(r["satisfaction_mean"], r["satisfaction_std"]))
            main = bold_if_best(r["mean"], best_main, fmt_mean_std(r["mean"], r["std"]))
            pref = bold_if_best(r["preferred_count"], best_pref, fmt_count_ratio(r["preferred_count"], r["total_trials"], r["preferred_ratio"]))
            ratio = bold_if_best(r["preferred_ratio"], best_ratio, fmt_percent(r["preferred_ratio"]))

            lines.append(
                f"| {r['method']} | {int(r['n'])} | {fit} | {help_} | {sat} | {main} | {pref} | {ratio} |"
            )

        lines.append("")
        lines.append(
            f"表注：表中评分为 1–7 分，数值越高表示参与者越认可该音乐推荐结果。"
            f"综合评分为符合性、帮助性和满意度三个维度的平均值。"
            f"最优选择比例表示该方法在每个 trial 的三种匿名推荐结果中被用户选为最愿意使用音乐的比例。"
            f"当前结果基于 {participant_count} 名参与者、{trial_count} 个情绪调节场景。"
        )

    return "\n".join(lines)


def make_main_table_en(stats, mode, participant_count, trial_count):
    lines = []

    if mode == "single":
        lines.append("## Table 2. Main results of the user study")
        lines.append("")
        lines.append("| Method | N | Regulation suitability ↑ | Median | Preferred count ↑ | Preference ratio ↑ |")
        lines.append("|---|---:|---:|---:|---:|---:|")

        best_mean = stats["mean"].max()
        best_median = stats["median"].max()
        best_pref = stats["preferred_count"].max()
        best_ratio = stats["preferred_ratio"].max()

        for _, r in stats.iterrows():
            score = bold_if_best(r["mean"], best_mean, fmt_mean_std(r["mean"], r["std"]))
            median = bold_if_best(r["median"], best_median, f"{r['median']:.2f}")
            pref = bold_if_best(r["preferred_count"], best_pref, fmt_count_ratio(r["preferred_count"], r["total_trials"], r["preferred_ratio"]))
            ratio = bold_if_best(r["preferred_ratio"], best_ratio, fmt_percent(r["preferred_ratio"]))

            lines.append(
                f"| {r['method']} | {int(r['n'])} | {score} | {median} | {pref} | {ratio} |"
            )

        lines.append("")
        lines.append(
            f"Note: Regulation suitability is rated on a 1–7 scale, where higher scores indicate that participants considered the music more suitable for helping them move from the current emotional state to the target state. "
            f"The preference ratio denotes how often each method was selected as the preferred option among the three anonymous recommendations in each trial. "
            f"The results are based on {participant_count} participants and {trial_count} emotion-regulation scenarios."
        )

    else:
        lines.append("## Table 2. Main results of the user study")
        lines.append("")
        lines.append("| Method | N | Fit ↑ | Helpfulness ↑ | Satisfaction ↑ | Overall score ↑ | Preferred count ↑ | Preference ratio ↑ |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

        best_fit = stats["regulation_fit_mean"].max()
        best_help = stats["helpfulness_mean"].max()
        best_sat = stats["satisfaction_mean"].max()
        best_main = stats["mean"].max()
        best_pref = stats["preferred_count"].max()
        best_ratio = stats["preferred_ratio"].max()

        for _, r in stats.iterrows():
            fit = bold_if_best(r["regulation_fit_mean"], best_fit, fmt_mean_std(r["regulation_fit_mean"], r["regulation_fit_std"]))
            help_ = bold_if_best(r["helpfulness_mean"], best_help, fmt_mean_std(r["helpfulness_mean"], r["helpfulness_std"]))
            sat = bold_if_best(r["satisfaction_mean"], best_sat, fmt_mean_std(r["satisfaction_mean"], r["satisfaction_std"]))
            main = bold_if_best(r["mean"], best_main, fmt_mean_std(r["mean"], r["std"]))
            pref = bold_if_best(r["preferred_count"], best_pref, fmt_count_ratio(r["preferred_count"], r["total_trials"], r["preferred_ratio"]))
            ratio = bold_if_best(r["preferred_ratio"], best_ratio, fmt_percent(r["preferred_ratio"]))

            lines.append(
                f"| {r['method']} | {int(r['n'])} | {fit} | {help_} | {sat} | {main} | {pref} | {ratio} |"
            )

        lines.append("")
        lines.append(
            f"Note: All scores are rated on a 1–7 scale, where higher scores indicate stronger subjective agreement. "
            f"The overall score is the average of fit, helpfulness, and satisfaction. "
            f"The preference ratio denotes how often each method was selected as the preferred option among the three anonymous recommendations in each trial. "
            f"The results are based on {participant_count} participants and {trial_count} emotion-regulation scenarios."
        )

    return "\n".join(lines)


def make_strategy_table_zh(strategy_stats):
    if strategy_stats.empty:
        return ""

    lines = []
    lines.append("## 按调节策略分组的附表")
    lines.append("")
    lines.append("| 调节策略 | 方法 | 评分数 N | 调节适配评分 ↑ | 被选为最优次数 ↑ | 最优选择比例 ↑ |")
    lines.append("|---|---|---:|---:|---:|---:|")

    for strategy in sorted(strategy_stats["strategy"].unique()):
        sub_strategy = strategy_stats[strategy_stats["strategy"] == strategy]
        best_mean = sub_strategy["mean"].max()
        best_pref = sub_strategy["preferred_count"].max()
        best_ratio = sub_strategy["preferred_ratio"].max()

        for _, r in sub_strategy.iterrows():
            score = bold_if_best(r["mean"], best_mean, fmt_mean_std(r["mean"], r["std"]))
            pref = bold_if_best(r["preferred_count"], best_pref, fmt_count_ratio(r["preferred_count"], r["total_trials"], r["preferred_ratio"]))
            ratio = bold_if_best(r["preferred_ratio"], best_ratio, fmt_percent(r["preferred_ratio"]))

            lines.append(
                f"| {strategy} | {r['method']} | {int(r['n'])} | {score} | {pref} | {ratio} |"
            )

    return "\n".join(lines)


def make_strategy_table_en(strategy_stats):
    if strategy_stats.empty:
        return ""

    lines = []
    lines.append("## Supplementary table by regulation strategy")
    lines.append("")
    lines.append("| Strategy | Method | N | Regulation suitability ↑ | Preferred count ↑ | Preference ratio ↑ |")
    lines.append("|---|---|---:|---:|---:|---:|")

    for strategy in sorted(strategy_stats["strategy"].unique()):
        sub_strategy = strategy_stats[strategy_stats["strategy"] == strategy]
        best_mean = sub_strategy["mean"].max()
        best_pref = sub_strategy["preferred_count"].max()
        best_ratio = sub_strategy["preferred_ratio"].max()

        for _, r in sub_strategy.iterrows():
            score = bold_if_best(r["mean"], best_mean, fmt_mean_std(r["mean"], r["std"]))
            pref = bold_if_best(r["preferred_count"], best_pref, fmt_count_ratio(r["preferred_count"], r["total_trials"], r["preferred_ratio"]))
            ratio = bold_if_best(r["preferred_ratio"], best_ratio, fmt_percent(r["preferred_ratio"]))

            lines.append(
                f"| {strategy} | {r['method']} | {int(r['n'])} | {score} | {pref} | {ratio} |"
            )

    return "\n".join(lines)


def save_outputs(stats, strategy_stats, mode, participant_count, trial_count, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    zh = make_main_table_zh(stats, mode, participant_count, trial_count)
    en = make_main_table_en(stats, mode, participant_count, trial_count)

    strategy_zh = make_strategy_table_zh(strategy_stats)
    strategy_en = make_strategy_table_en(strategy_stats)

    if strategy_zh:
        zh = zh + "\n\n" + strategy_zh

    if strategy_en:
        en = en + "\n\n" + strategy_en

    zh_path = out_dir / "paper_main_table_zh.md"
    en_path = out_dir / "paper_main_table_en.md"
    csv_path = out_dir / "paper_main_table_stats.csv"

    zh_path.write_text(zh, encoding="utf-8")
    en_path.write_text(en, encoding="utf-8")
    stats.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"评分模式: {mode}")
    print(f"参与者数量: {participant_count}")
    print(f"trial 数量: {trial_count}")
    print(f"中文表格: {zh_path}")
    print(f"英文表格: {en_path}")
    print(f"统计 CSV: {csv_path}")
    print("=" * 80)
    print(zh)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Chinese and English paper tables from responses_merged.csv."
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to responses_merged.csv",
    )

    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="",
        help="Output directory. Default: same folder as input CSV / paper_tables",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out) if args.out else input_path.parent / "paper_tables"

    df = read_csv_safely(input_path)
    mode = detect_score_mode(df)
    df = prepare_scores(df, mode)
    df = prepare_preference(df)

    participant_count, trial_count = participant_trial_count(df)

    stats = aggregate_main_table(df, mode)
    strategy_stats = aggregate_strategy_table(df)

    save_outputs(
        stats=stats,
        strategy_stats=strategy_stats,
        mode=mode,
        participant_count=participant_count,
        trial_count=trial_count,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()