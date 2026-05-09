from __future__ import annotations

import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parents[2]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st

from application.config import strength, strategies
from application.web.plots import make_va_plot
from application.web.service import recommend
from application.web.ui import (
    render_recommendations,
    render_sidebar,
    render_state_summary,
)


def main():
    st.set_page_config(
        page_title="Emotion-Regulation Music Recommendation Demo",
        layout="wide",
    )

    st.title("情绪调节型音乐推荐")
    st.caption("输入你现在的状态，选择希望音乐带来的帮助，系统会推荐更合适的音乐。")

    (
        text,
        strategy,
        strength_value,
        top_k,
        run_btn,
    ) = render_sidebar(strategies, strength)

    if not run_btn:
        st.info("在左侧输入当前状态，选择调节方向，然后点击“开始推荐”。")
        return

    with st.spinner("正在理解你的状态并检索音乐..."):
        state, recs, all_results, _ = recommend(
            text=text,
            strategy=strategy,
            strength=strength_value,
            gamma=1.0,
            top_k=top_k,
            diversity_mode="关闭",
            prior_mode="prototype",
        )

    selected = recs["Ours"]

    render_state_summary(state)

    st.subheader("2. 推荐结果")
    st.caption("下面的音乐综合考虑了你当前的状态、希望达到的调节方向，以及音乐本身的情绪与标签特征。")

    render_recommendations(
        f"Top-{top_k} 推荐",
        selected,
        show_table=False,
    )

    st.subheader("3. 情绪空间")
    st.caption("图中展示了当前情绪、目标方向和推荐音乐在情绪空间中的位置。")

    fig = make_va_plot(
        all_tracks=all_results,
        selected_top=selected,
        z_q=state["z_q"],
        z_anchor=state["anchor"],
        mu_q=state["mu_q"],
        label="推荐音乐",
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
