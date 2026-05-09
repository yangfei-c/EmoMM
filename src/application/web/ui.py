from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


STRATEGY_TEXT = {
    "maintain": "保持当前情绪",
    "calm": "平静下来",
    "comfort": "获得安慰",
    "energize": "提升能量",
}

STRATEGY_HELP = {
    "maintain": "尽量保持当前状态，不主动拉高或降低情绪。",
    "calm": "降低唤醒度，让状态逐渐平静、稳定。",
    "comfort": "在保留一定同质性的同时，提供陪伴和安慰。",
    "energize": "提高唤醒度，让状态更积极、更有行动感。",
}


def describe_va(z):
    v, a = float(z[0]), float(z[1])
    valence = "偏积极" if v >= 0 else "偏消极"
    arousal = "高唤醒" if a >= 0 else "低唤醒"
    return f"{valence} / {arousal}"


def render_sidebar(strategies, default_strength):
    with st.sidebar:
        st.header("输入设置")

        text = st.text_area(
            "你现在的状态",
            value="我准备去跑步了",
            height=120,
            help="输入一句描述当前情绪或场景的话。",
        )

        available_strategy_labels = [
            f"{STRATEGY_TEXT.get(s, s)}" for s in strategies
        ]

        default_strategy = "energize" if "energize" in strategies else strategies[0]
        default_strategy_index = strategies.index(default_strategy)

        strategy_label = st.selectbox(
            "希望音乐怎么帮助你？",
            available_strategy_labels,
            index=default_strategy_index,
        )
        strategy = strategies[available_strategy_labels.index(strategy_label)]

        st.caption(STRATEGY_HELP.get(strategy, ""))

        strength = st.slider(
            "调节强度",
            0.0,
            1.0,
            float(default_strength),
            0.05,
            help="数值越大，推荐越倾向于目标调节方向。",
        )

        top_k = st.slider(
            "推荐数量",
            3,
            10,
            5,
            1,
        )

        run_btn = st.button("开始推荐", type="primary", use_container_width=True)

    return (
        text,
        strategy,
        strength,
        top_k,
        run_btn,
    )


def render_state_summary(state):
    st.subheader("1. 情绪状态与调节方向")

    c1, c2, c3 = st.columns(3)

    c1.metric("当前状态", describe_va(state["z_q"]))
    c2.metric("目标方向", describe_va(state["mu_q"]))
    c3.metric("调节方式", STRATEGY_TEXT.get(state["strategy"], state["strategy"]))

    selected_labels = state.get("selected_strategy_tags", [])

    st.caption("系统会结合当前状态和目标方向，推荐情绪过渡更自然、语义更匹配的音乐。")

    if selected_labels:
        st.markdown(
            "**相关音乐标签：** "
            + " · ".join(f"`{x}`" for x in selected_labels)
        )


def to_df(items):
    rows = []
    for rank, item in enumerate(items, start=1):
        rows.append({
            "Rank": rank,
            "Song ID": item["song_id"],
            "Valence": round(item["V"], 4),
            "Arousal": round(item["A"], 4),
            "Tags": item["display_tags"],
        })
    return pd.DataFrame(rows)


def make_reason(item):
    tags = item.get("display_tags", "")
    return f"推荐理由：这首歌的情绪位置较适合当前调节目标，且音乐标签包含 {tags}。"


def render_recommendations(title, items, show_table=False):
    st.markdown(f"### {title}")

    if show_table:
        st.dataframe(to_df(items), use_container_width=True, hide_index=True)

    for rank, item in enumerate(items, start=1):
        path = Path(item["song_path"])

        with st.container(border=True):
            st.markdown(f"#### 推荐 {rank} · {item['song_id']}")
            st.caption(make_reason(item))

            st.markdown(
                f"**标签：** {item['display_tags']}  \n"
                f"**情绪位置：** [{item['V']:.4f}, {item['A']:.4f}]"
            )

            if path.exists():
                st.audio(str(path), format="audio/mpeg")
            else:
                st.warning(f"音频文件不存在：{path}")
