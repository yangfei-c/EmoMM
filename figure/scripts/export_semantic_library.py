from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
SOURCE_JSON = ROOT / "data" / "music_db" / "semantic_db.json"
OUTPUT_DIR = ROOT / "figure" / "data" / "02_semantic_library"
OUTPUT_CSV = OUTPUT_DIR / "semantic.csv"
OUTPUT_MD = OUTPUT_DIR / "semantic.md"

TAGS_26 = [
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

GENRE_TAGS = TAGS_26[:10]
INSTRUMENT_TAGS = TAGS_26[10:17]
MOOD_THEME_TAGS = TAGS_26[17:]

CSV_HEADER = [
    "song_id",
    "valence",
    "arousal",
    "tag_ambient",
    "tag_classical",
    "tag_electronic",
    "tag_folk",
    "tag_hiphop",
    "tag_jazz",
    "tag_metal",
    "tag_pop",
    "tag_rock",
    "tag_soundtrack",
    "tag_bass",
    "tag_drums",
    "tag_guitar",
    "tag_piano",
    "tag_strings",
    "tag_synthesizer",
    "tag_voice",
    "tag_dark",
    "tag_dream",
    "tag_energetic",
    "tag_epic",
    "tag_film",
    "tag_happy",
    "tag_love",
    "tag_relaxing",
    "tag_sad",
    "top5_tags",
    "top3_genre_tags",
    "top3_instrument_tags",
    "top3_mood/theme_tags",
]


def format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def topk_tags(tag_scores: list[float], tag_names: list[str], k: int) -> list[str]:
    pairs = sorted(zip(tag_scores, tag_names), key=lambda x: x[0], reverse=True)
    return [name for _, name in pairs[:k]]


def get_tag_scores(vec: list[float]) -> list[float] | None:
    if len(vec) < 28:
        return None
    return vec[2:28]


def row_from_entry(entry: dict) -> list[str] | None:
    song_id = entry.get("song_id")
    vec = entry.get("semantic_vector_28d")
    if song_id is None or not isinstance(vec, list):
        return None
    tag_scores = get_tag_scores(vec)
    if tag_scores is None or len(tag_scores) != 26:
        return None

    valence = float(vec[0])
    arousal = float(vec[1])

    top5 = topk_tags(tag_scores, TAGS_26, 5)
    top3_genre = topk_tags([tag_scores[TAGS_26.index(t)] for t in GENRE_TAGS], GENRE_TAGS, 3)
    top3_instrument = topk_tags([tag_scores[TAGS_26.index(t)] for t in INSTRUMENT_TAGS], INSTRUMENT_TAGS, 3)
    top3_mood_theme = topk_tags([tag_scores[TAGS_26.index(t)] for t in MOOD_THEME_TAGS], MOOD_THEME_TAGS, 3)

    row = [
        str(song_id),
        format_float(valence),
        format_float(arousal),
    ]
    row.extend(format_float(float(score)) for score in tag_scores)
    row.extend(
        [
            ",".join(top5),
            ",".join(top3_genre),
            ",".join(top3_instrument),
            ",".join(top3_mood_theme),
        ]
    )
    return row


def load_entries() -> Iterable[dict]:
    data = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.values()
    return []


def write_semantic_csv(rows: list[list[str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def build_tag_table_md() -> list[str]:
    lines = ["| 序号 | 字段名 | 标签名 |", "| --- | --- | --- |"]
    for idx, name in enumerate(TAGS_26, start=1):
        lines.append(f"| {idx} | tag_{name} | {name} |")
    return lines


def build_sample_table_md(rows: list[list[str]], limit: int = 10) -> list[str]:
    lines = ["| song_id | valence | arousal | top5_tags |", "| --- | --- | --- | --- |"]
    for row in rows[:limit]:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[29]} |")
    return lines


def write_semantic_md(exported: int, skipped: int, rows: list[list[str]]) -> None:
    lines: list[str] = []
    lines.append("# 音乐语义库导出结果")
    lines.append("")
    lines.append("## 一、音乐语义库结构")
    lines.append("说明每首音乐由 2 维 Music VA 预测值和 26 维 Music Tag 预测概率构成，共 28 维情绪—语义表示。")
    lines.append(f"本次导出成功 {exported} 条，跳过 {skipped} 条（缺少 VA 或 26 维 Tag）。")
    lines.append("")
    lines.append("## 二、26 个具体标签")
    lines.append("")
    lines.extend(build_tag_table_md())
    lines.append("")
    lines.append("## 三、语义库样例")
    lines.append("")
    lines.extend(build_sample_table_md(rows))
    lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows: list[list[str]] = []
    skipped = 0
    for entry in load_entries():
        row = row_from_entry(entry)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    write_semantic_csv(rows)
    write_semantic_md(len(rows), skipped, rows)

    print(f"semantic.csv: {OUTPUT_CSV}")
    print(f"semantic.md: {OUTPUT_MD}")
    print(f"exported: {len(rows)}")
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
