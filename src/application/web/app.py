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
from application.web.ui import render_recommendations, render_sidebar, render_state


FORMULAS = {
    "VA-only": "$$S_{VA}(i)=-\\|z_i-\\mu_q\\|_2^2$$",
    "Tag-only": "$$S_{Tag}(i)=y_i^\\top\\tau_s$$",
    "VA + Tag": "$$S(i|q,s)=-\\|z_i-\\mu_q\\|_2^2+\\gamma y_i^\\top\\tau_s$$",
}


def main():
    st.set_page_config(
        page_title="Emotion-Regulation Music Recommendation Demo",
        layout="wide",
    )

    st.title("情绪调节型音乐推荐 Demo")
    st.caption("Text VA → Strategy Anchor → Target VA → VA/Tag Joint Ranking")

    (
        text,
        strategy,
        strength,
        gamma,
        top_k,
        ranking_mode,
        diversity_mode,
        show_debug,
        run_btn,
    ) = render_sidebar(strategies, strength)

    if not run_btn:
        st.info("在左侧输入文本、策略和强度，然后点击“开始推荐”。")
        return

    with st.spinner("正在预测文本 VA 并检索音乐..."):
        state, recs, all_results, diversity_weight = recommend(
            text=text,
            strategy=strategy,
            strength=strength,
            gamma=gamma,
            top_k=top_k,
            diversity_mode=diversity_mode,
        )

    render_state(state, diversity_mode, diversity_weight)

    st.subheader("2. VA 空间可视化")
    selected = recs[ranking_mode]
    fig = make_va_plot(
        all_tracks=all_results,
        selected_top=selected,
        z_q=state["z_q"],
        z_anchor=state["anchor"],
        mu_q=state["mu_q"],
        label=f"{ranking_mode} Top-K",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3. 推荐结果")
    st.markdown(FORMULAS[ranking_mode])
    render_recommendations(
        f"{ranking_mode} Top-K",
        selected,
        show_debug=show_debug,
    )

    if show_debug:
        st.divider()
        st.subheader("4. 三种排序结果对比")
        st.caption("以下结果均已应用当前标签多样性设置。")
        for mode in ["VA-only", "Tag-only", "VA + Tag"]:
            with st.expander(mode, expanded=(mode == ranking_mode)):
                st.markdown(FORMULAS[mode])
                render_recommendations(
                    f"{mode} Top-K",
                    recs[mode],
                    show_debug=True,
                )


if __name__ == "__main__":
    main()
