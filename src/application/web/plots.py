from __future__ import annotations

import plotly.graph_objects as go


def make_va_plot(all_tracks, selected_top, z_q, z_anchor, mu_q, label="Recommendation TopK"):
    bg = all_tracks[:: max(1, len(all_tracks) // 2000)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[t["V"] for t in bg],
        y=[t["A"] for t in bg],
        mode="markers",
        name="Music library",
        marker=dict(size=4, opacity=0.15),
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=[t["V"] for t in selected_top],
        y=[t["A"] for t in selected_top],
        mode="markers",
        name=label,
        marker=dict(size=11, symbol="star"),
        text=[f'{t["song_id"]}<br>{t["display_tags"]}' for t in selected_top],
        hoverinfo="text+x+y",
    ))

    fig.add_trace(go.Scatter(
        x=[z_q[0]],
        y=[z_q[1]],
        mode="markers+text",
        name="Current VA",
        text=["z_q"],
        textposition="top center",
        marker=dict(size=14, symbol="x"),
    ))

    fig.add_trace(go.Scatter(
        x=[z_anchor[0]],
        y=[z_anchor[1]],
        mode="markers+text",
        name="Strategy anchor",
        text=["z_s"],
        textposition="top center",
        marker=dict(size=14, symbol="square"),
    ))

    fig.add_trace(go.Scatter(
        x=[mu_q[0]],
        y=[mu_q[1]],
        mode="markers+text",
        name="Target VA",
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
        height=620,
        legend=dict(orientation="h"),
    )

    return fig
