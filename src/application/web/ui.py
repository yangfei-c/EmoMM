from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from application.config import default_diversity, diversity_weights


def render_sidebar(strategies, default_strength):
    with st.sidebar:
        st.header("输入设置")

        text = st.text_area("用户文本", value="我准备去跑步了", height=120)
        strategy = st.selectbox("调节策略", strategies, index=strategies.index("energize"))
        strength = st.slider("调节强度 strength", 0.0, 1.0, float(default_strength), 0.05)
        gamma = st.slider("Tag 权重 gamma", 0.0, 3.0, 1.0, 0.1)
        top_k = st.slider("Top-K", 3, 20, 5, 1)
        ranking_mode = st.selectbox("推荐模式", ["VA + Tag", "VA-only", "Tag-only"], index=0)
        diversity_mode = st.selectbox(
            "标签多样性",
            list(diversity_weights.keys()),
            index=list(diversity_weights.keys()).index(default_diversity),
        )
        show_debug = st.checkbox("显示论文/调试信息", value=False)
        run_btn = st.button("开始推荐", type="primary")

    return text, strategy, strength, gamma, top_k, ranking_mode, diversity_mode, show_debug, run_btn


def render_state(state, diversity_mode, diversity_weight):
    st.subheader("1. 用户情绪状态与目标 VA")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current V", f"{state['z_q'][0]:.4f}")
    c2.metric("Current A", f"{state['z_q'][1]:.4f}")
    c3.metric("Target V", f"{state['mu_q'][0]:.4f}")
    c4.metric("Target A", f"{state['mu_q'][1]:.4f}")

    st.markdown(
        f"""
        - **Strategy**: `{state["strategy"]}`
        - **Strength**: `{state["strength"]}`
        - **Strategy anchor**: `{state["anchor"]}`
        - **标签多样性**: `{diversity_mode}` / `eta={diversity_weight:.2f}`

        $$
        \\mu_q=(1-r)z_q+r z_s
        $$
        """
    )


def to_df(items, show_debug=False):
    rows = []

    for rank, item in enumerate(items, start=1):
        row = {
            "Rank": rank,
            "Song ID": item["song_id"],
            "Valence": round(item["V"], 4),
            "Arousal": round(item["A"], 4),
            "Tags": item["display_tags"],
        }

        if show_debug:
            row.update({
                "VA score": round(item["va_score"], 4),
                "Tag score": round(item["tag_score"], 4),
                "Total score": round(item["both_score"], 4),
                "Path": item["song_path"],
            })

        rows.append(row)

    return pd.DataFrame(rows)


def render_recommendations(title, items, show_debug=False):
    st.markdown(f"### {title}")
    st.dataframe(to_df(items, show_debug=show_debug), use_container_width=True, hide_index=True)

    for rank, item in enumerate(items, start=1):
        path = Path(item["song_path"])

        with st.container(border=True):
            st.markdown(
                f"**Top {rank}. {item['song_id']}**  \n"
                f"VA = [{item['V']:.4f}, {item['A']:.4f}]  \n"
                f"Tags = {item['display_tags']}"
            )

            if show_debug:
                st.caption(
                    f"VA score={item['va_score']:.4f} | "
                    f"Tag score={item['tag_score']:.4f} | "
                    f"Total={item['both_score']:.4f} | "
                    f"Path={item['song_path']}"
                )

            if path.exists():
                st.audio(str(path), format="audio/mpeg")
            else:
                st.warning(f"音频文件不存在：{path}")
