from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

current_dir = Path(__file__).resolve().parents[1]
if current_dir not in sys.path:
    sys.path.insert(0,str(current_dir))
from application.config import (
    default_strength,
    strategies,
    strategy_anchors,
)
from application.music_repository import load_tracks
from application.user_state import UserStateEstimator
from application.utils import va_score, tag_score, total_score


@st.cache_resource
def get_estimator() -> UserStateEstimator:
    return UserStateEstimator()


@st.cache_data
def get_tracks():
    return load_tracks()


def build_candidates(mu_q, tau, gamma=1.0):
    tracks = get_tracks()
    results = []

    for track in tracks:
        z_i = track["va"]
        y_i = track["tag26"]

        v_score = va_score(z_i, mu_q)
        t_score = tag_score(y_i, tau)
        both_score = total_score(z_i, y_i, mu_q, tau, gamma=gamma)

        results.append({
            "song_id": track["song_id"],
            "song_path": track["song_path"],
            "display_tags": ", ".join(track["display_tags"]),
            "va": z_i,
            "V": float(z_i[0]),
            "A": float(z_i[1]),
            "va_score": float(v_score),
            "tag_score": float(t_score),
            "both_score": float(both_score),
        })

    return results


def topk_by(results, key, top_k=5):
    return sorted(results, key=lambda x: x[key], reverse=True)[:top_k]


def to_df(items):
    rows = []
    for rank, item in enumerate(items, start=1):
        rows.append({
            "rank": rank,
            "song_id": item["song_id"],
            "V": round(item["V"], 4),
            "A": round(item["A"], 4),
            "display_tags": item["display_tags"],
            "VA score": round(item["va_score"], 4),
            "Tag score": round(item["tag_score"], 4),
            "VA+Tag score": round(item["both_score"], 4),
            "song_path": item["song_path"],
        })
    return pd.DataFrame(rows)


def make_va_plot(all_tracks, va_top, tag_top, both_top, z_q, z_anchor, mu_q):
    # 为了网页速度，只抽样展示音乐库背景点
    bg = all_tracks[:: max(1, len(all_tracks) // 2000)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[t["V"] for t in bg],
        y=[t["A"] for t in bg],
        mode="markers",
        name="Music library",
        marker=dict(size=4, opacity=0.18),
        hoverinfo="skip",
    ))

    def add_recs(items, name, symbol):
        fig.add_trace(go.Scatter(
            x=[t["V"] for t in items],
            y=[t["A"] for t in items],
            mode="markers",
            name=name,
            marker=dict(size=10, symbol=symbol),
            text=[f'{t["song_id"]}<br>{t["display_tags"]}' for t in items],
            hoverinfo="text+x+y",
        ))

    add_recs(va_top, "VA-only TopK", "circle")
    add_recs(tag_top, "Tag-only TopK", "diamond")
    add_recs(both_top, "VA+Tag TopK", "star")

    fig.add_trace(go.Scatter(
        x=[z_q[0]],
        y=[z_q[1]],
        mode="markers+text",
        name="Current VA z_q",
        text=["z_q"],
        textposition="top center",
        marker=dict(size=14, symbol="x"),
    ))

    fig.add_trace(go.Scatter(
        x=[z_anchor[0]],
        y=[z_anchor[1]],
        mode="markers+text",
        name="Strategy anchor z_s",
        text=["z_s"],
        textposition="top center",
        marker=dict(size=14, symbol="square"),
    ))

    fig.add_trace(go.Scatter(
        x=[mu_q[0]],
        y=[mu_q[1]],
        mode="markers+text",
        name="Target VA μ_q",
        text=["μ_q"],
        textposition="top center",
        marker=dict(size=16, symbol="cross"),
    ))

    fig.add_annotation(
        x=mu_q[0],
        y=mu_q[1],
        ax=z_q[0],
        ay=z_q[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.5,
        arrowwidth=2,
    )

    fig.update_layout(
        title="Valence-Arousal Space",
        xaxis_title="Valence",
        yaxis_title="Arousal",
        xaxis=dict(range=[-1, 1], zeroline=True),
        yaxis=dict(range=[-1, 1], zeroline=True),
        height=650,
        legend=dict(orientation="h"),
    )

    return fig


def show_audio_list(title, items):
    st.markdown(f"### {title}")
    df = to_df(items)
    st.dataframe(df, use_container_width=True)

    with st.expander(f"播放 {title} 的 Top 音乐"):
        for item in items:
            path = Path(item["song_path"])
            st.markdown(
                f"**{item['song_id']}**  \n"
                f"VA = [{item['V']:.4f}, {item['A']:.4f}]  \n"
                f"Tags = {item['display_tags']}  \n"
                f"Score = {item['both_score']:.4f}"
            )
            if path.exists():
                st.audio(str(path), format="audio/mpeg")
            else:
                st.warning(f"音频文件不存在：{path}")


def main():
    st.set_page_config(
        page_title="Emotion-Regulation Music Recommendation Demo",
        layout="wide",
    )

    st.title("情绪调节型音乐推荐 Demo")
    st.caption("Text VA → Strategy Anchor → Target VA → VA/Tag Joint Ranking")

    with st.sidebar:
        st.header("输入设置")

        text = st.text_area(
            "用户文本",
            value="I am sad",
            height=120,
        )

        strategy = st.selectbox(
            "调节策略",
            strategies,
            index=strategies.index("comfort") if "comfort" in strategies else 0,
        )

        strength = st.slider(
            "调节强度 strength",
            min_value=0.0,
            max_value=1.0,
            value=float(default_strength),
            step=0.05,
        )

        gamma = st.slider(
            "Tag 权重 gamma",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
        )

        top_k = st.slider(
            "Top-K",
            min_value=3,
            max_value=20,
            value=5,
            step=1,
        )

        run_btn = st.button("开始推荐", type="primary")

    if not run_btn:
        st.info("在左侧输入文本、策略和强度，然后点击“开始推荐”。")
        return

    estimator = get_estimator()

    with st.spinner("正在预测文本 VA 并检索音乐..."):
        state = estimator.process(text, strategy=strategy, strength=strength)
        mu_q = state["mu_q"]
        tau = state["tag_prior"]

        results = build_candidates(mu_q, tau, gamma=gamma)
        va_top = topk_by(results, "va_score", top_k=top_k)
        tag_top = topk_by(results, "tag_score", top_k=top_k)
        both_top = topk_by(results, "both_score", top_k=top_k)

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
        - **Target VA**:

        $$
        \\mu_q=(1-r)z_q+r z_s
        $$
        """
    )

    st.subheader("2. VA 空间可视化")
    fig = make_va_plot(
        all_tracks=results,
        va_top=va_top,
        tag_top=tag_top,
        both_top=both_top,
        z_q=state["z_q"],
        z_anchor=state["anchor"],
        mu_q=state["mu_q"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3. 三种推荐结果对比")

    tab1, tab2, tab3 = st.tabs(["VA-only", "Tag-only", "VA + Tag"])

    with tab1:
        st.markdown(
            "$$S_{VA}(i)=-\\|z_i-\\mu_q\\|_2^2$$\n\n"
            "只看音乐 VA 是否接近目标 VA。"
        )
        show_audio_list("VA-only Top-K", va_top)

    with tab2:
        st.markdown(
            "$$S_{Tag}(i)=y_i^\\top\\tau_s$$\n\n"
            "只看音乐标签是否符合策略语义。"
        )
        show_audio_list("Tag-only Top-K", tag_top)

    with tab3:
        st.markdown(
            "$$S(i|q,s)=-\\|z_i-\\mu_q\\|_2^2+\\gamma y_i^\\top\\tau_s$$\n\n"
            "同时考虑目标 VA 匹配与策略标签匹配。"
        )
        show_audio_list("VA+Tag Top-K", both_top)


if __name__ == "__main__":
    main()