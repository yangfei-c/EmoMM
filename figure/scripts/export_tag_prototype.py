from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_JSON = ROOT / "data" / "music_db" / "semantic_db.json"
OUTPUT_DIR = ROOT / "figure" / "data" / "03_tag_prototype"
OUTPUT_CSV = OUTPUT_DIR / "tag_prototype.csv"
OUTPUT_MD = OUTPUT_DIR / "tag_prototype.md"

TAGS = [
    "ambient",
    "classical",
    "electronic",
    "folk",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "rock",
    "soundtrack",
    "bass",
    "drums",
    "guitar",
    "piano",
    "strings",
    "synthesizer",
    "voice",
    "dark",
    "dream",
    "energetic",
    "epic",
    "film",
    "happy",
    "love",
    "relaxing",
    "sad",
]

GROUPS = {
    "genre": TAGS[:10],
    "instrument": TAGS[10:17],
    "mood": TAGS[17:],
}

STRATEGY_TAGS = {
    "calm": ["ambient", "relaxing", "piano", "dream", "classical"],
    "comfort": ["piano", "sad", "classical", "dream", "relaxing"],
    "energize": ["energetic", "electronic", "rock", "pop", "happy"],
}

CSV_HEADER = [
    "tag",
    "group",
    "full_valence",
    "full_arousal",
    "top5_valence",
    "top5_arousal",
    "delta_valence",
    "delta_arousal",
    "top5_n",
    "used_for_strategy",
]


def load_vectors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        entries: Iterable[dict] = data.values()
    elif isinstance(data, list):
        entries = data
    else:
        entries = []

    vecs = [
        x["semantic_vector_28d"]
        for x in entries
        if isinstance(x, dict)
        and isinstance(x.get("semantic_vector_28d"), list)
        and len(x["semantic_vector_28d"]) == 28
    ]
    arr = np.asarray(vecs, dtype=float)
    return arr[:, :2], arr[:, 2:]


def weighted_proto(va: np.ndarray, probs: np.ndarray, top_ratio: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    protos = np.zeros((probs.shape[1], 2), dtype=float)
    top_counts = np.zeros(probs.shape[1], dtype=int)
    for j in range(probs.shape[1]):
        w = probs[:, j]
        z = va
        if top_ratio is not None:
            thr = float(np.quantile(w, 1 - top_ratio))
            mask = w >= thr
            w = w[mask]
            z = z[mask]
            top_counts[j] = int(mask.sum())
        denom = float(w.sum()) + 1e-12
        if denom <= 1e-12 or len(w) == 0:
            protos[j] = np.array([np.nan, np.nan])
            continue
        protos[j] = (w[:, None] * z).sum(axis=0) / denom
    return protos, top_counts


def tag_group(tag: str) -> str:
    for name, tags in GROUPS.items():
        if tag in tags:
            return name
    return ""


def used_for_strategy(tag: str) -> str:
    matches = [name for name, tags in STRATEGY_TAGS.items() if tag in tags]
    return "|".join(matches)


def fmt_pair(valence: float, arousal: float) -> str:
    if np.isnan(valence) or np.isnan(arousal):
        return "-"
    return f"({valence:.4f}, {arousal:.4f})"


def write_csv(rows: list[list[str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def build_md(rows: list[dict], anchors: dict[str, tuple[float, float]]) -> str:
    lines: list[str] = []
    lines.append("# Tag 原型导出结果")
    lines.append("")
    lines.append("## 一、Tag 原型构建方式")
    lines.append("- full 原型使用全部音乐样本；")
    lines.append("- top5 原型使用每个标签预测概率最高的前 5% 音乐样本；")
    lines.append("- VA 坐标来自 Music VA 模型预测属性；")
    lines.append("- Tag 权重来自 Music Tag 模型预测概率；")
    lines.append("- Tag 原型是基于模型预测伪属性构建的统计原型，不是真实人工标签。")
    lines.append("")
    lines.append("## 二、全量样本原型与 Top-5% 原型对比")
    lines.append("")
    lines.append("| Tag | Group | Full VA | Top-5% VA | ΔVA | 策略用途 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            "| {tag} | {group} | {full} | {top5} | {delta} | {strategy} |".format(
                tag=row["tag"],
                group=row["group"],
                full=fmt_pair(row["full_valence"], row["full_arousal"]),
                top5=fmt_pair(row["top5_valence"], row["top5_arousal"]),
                delta=fmt_pair(row["delta_valence"], row["delta_arousal"]),
                strategy=row["used_for_strategy"] or "-",
            )
        )
    lines.append("")
    lines.append("## 三、四类调节策略与代表标签")
    lines.append("")
    lines.append("| 策略 | 代表标签 | 策略锚点 VA | 说明 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        "| calm | ambient, relaxing, piano, dream, classical | {va} | top5 原型平均 |".format(
            va=fmt_pair(*anchors["calm"])
        )
    )
    lines.append(
        "| comfort | piano, sad, classical, dream, relaxing | {va} | top5 原型平均 |".format(
            va=fmt_pair(*anchors["comfort"])
        )
    )
    lines.append(
        "| energize | energetic, electronic, rock, pop, happy | {va} | top5 原型平均 |".format(
            va=fmt_pair(*anchors["energize"])
        )
    )
    lines.append("| maintain | - | a_r = z_q | 动态策略，不固定锚点 |")
    lines.append("")
    return "\n".join(lines)


def compute_anchors(top5: np.ndarray) -> dict[str, tuple[float, float]]:
    tag_to_idx = {tag: idx for idx, tag in enumerate(TAGS)}
    anchors: dict[str, tuple[float, float]] = {}
    for name, tags in STRATEGY_TAGS.items():
        coords = np.array([top5[tag_to_idx[tag]] for tag in tags], dtype=float)
        anchors[name] = (float(np.nanmean(coords[:, 0])), float(np.nanmean(coords[:, 1])))
    return anchors


def main() -> None:
    va, probs = load_vectors(SOURCE_JSON)
    full, _ = weighted_proto(va, probs)
    top5, top5_counts = weighted_proto(va, probs, top_ratio=0.05)

    rows_csv: list[list[str]] = []
    rows_md: list[dict] = []
    for idx, tag in enumerate(TAGS):
        full_v, full_a = full[idx]
        top_v, top_a = top5[idx]
        delta_v = top_v - full_v
        delta_a = top_a - full_a
        top_n = int(top5_counts[idx])
        strategy = used_for_strategy(tag)
        group = tag_group(tag)

        rows_csv.append(
            [
                tag,
                group,
                f"{full_v:.6f}",
                f"{full_a:.6f}",
                f"{top_v:.6f}",
                f"{top_a:.6f}",
                f"{delta_v:.6f}",
                f"{delta_a:.6f}",
                str(top_n),
                strategy,
            ]
        )
        rows_md.append(
            {
                "tag": tag,
                "group": group,
                "full_valence": float(full_v),
                "full_arousal": float(full_a),
                "top5_valence": float(top_v),
                "top5_arousal": float(top_a),
                "delta_valence": float(delta_v),
                "delta_arousal": float(delta_a),
                "used_for_strategy": strategy,
            }
        )

    anchors = compute_anchors(top5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows_csv)
    OUTPUT_MD.write_text(build_md(rows_md, anchors), encoding="utf-8")

    print(f"tag_prototype.csv: {OUTPUT_CSV}")
    print(f"tag_prototype.md: {OUTPUT_MD}")
    print(f"computed_tags: {len(TAGS)}")
    for tag, count in zip(TAGS, top5_counts):
        print(f"top5_n[{tag}]: {int(count)}")
    print(
        "anchors: calm=({:.6f},{:.6f}), comfort=({:.6f},{:.6f}), energize=({:.6f},{:.6f})".format(
            anchors["calm"][0],
            anchors["calm"][1],
            anchors["comfort"][0],
            anchors["comfort"][1],
            anchors["energize"][0],
            anchors["energize"][1],
        )
    )


if __name__ == "__main__":
    main()

