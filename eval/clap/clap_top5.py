# -*- coding: utf-8 -*-

import csv
import json
import sys
from pathlib import Path

import numpy as np
from msclap import CLAP


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")
sys.path.insert(0, str(ROOT / "src"))

from core.music_repository import load_tracks


QUERY_JSON = ROOT / "eval" / "queries_zh16.json"

CLAP_DIR = Path(r"D:\AAAA_audio_dataset\eval\clap")
AUDIO_EMB = CLAP_DIR / "audio_embd.npz"
TEXT_EMB = CLAP_DIR / "text_embd.npz"

OUT_CSV = ROOT / "eval" / "clap" / "clap_top5.csv"

TOP_K = 5


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_queries():
    return json.loads(QUERY_JSON.read_text(encoding="utf-8"))


def qid(q):
    return str(q.get("query_id") or q.get("id"))


def query_text(q):
    text = str(q.get("text") or "").strip()
    if text:
        return text

    text = f"{q.get('current_state', '')}{q.get('regulation_goal', '')}".strip()
    if not text:
        raise ValueError(f"{qid(q)} 缺少 text")
    return text


def need_rebuild_text_emb(query_ids, query_texts):
    if not TEXT_EMB.exists():
        return True

    data = np.load(TEXT_EMB, allow_pickle=True)

    if "query_ids" not in data or "query_texts" not in data or "text_emb" not in data:
        return True

    old_ids = [str(x) for x in data["query_ids"]]
    old_texts = [str(x) for x in data["query_texts"]]

    return old_ids != query_ids or old_texts != query_texts


def build_text_emb(model, query_ids, query_texts):
    vecs = []

    for i, text in enumerate(query_texts, start=1):
        emb = model.get_text_embeddings([text])
        vecs.append(to_numpy(emb).astype(np.float32))
        print(f"text embedding: {i}/{len(query_texts)}")

    text_emb = np.concatenate(vecs, axis=0)

    TEXT_EMB.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        TEXT_EMB,
        query_ids=np.array(query_ids),
        query_texts=np.array(query_texts),
        text_emb=text_emb,
    )

    return text_emb


def load_or_build_text_emb(query_ids, query_texts):
    model = CLAP(version="2023", use_cuda=True)

    if need_rebuild_text_emb(query_ids, query_texts):
        return build_text_emb(model, query_ids, query_texts)

    return np.load(TEXT_EMB, allow_pickle=True)["text_emb"].astype(np.float32)


def load_audio_emb():
    if not AUDIO_EMB.exists():
        raise FileNotFoundError(AUDIO_EMB)

    data = np.load(AUDIO_EMB, allow_pickle=True)

    return {
        "song_ids": [str(x) for x in data["song_ids"]],
        "audio_paths": [str(x) for x in data["audio_paths"]],
        "audio_emb": data["audio_emb"].astype(np.float32),
    }


def load_track_meta():
    return {str(t["song_id"]): t for t in load_tracks()}


def make_row(q, song_id, audio_path, score, rank, track):
    return {
        "method": "clap",
        "query_id": qid(q),
        "rank": rank,
        "song_id": song_id,
        "score": round(float(score), 6),

        "text": query_text(q),
        "current_state": str(q.get("current_state", "")).strip(),
        "regulation_goal": str(q.get("regulation_goal", "")).strip(),
        "strategy": q.get("strategy", ""),
        "strategy_zh": str(q.get("strategy_zh") or q.get("strategy") or "").strip(),
        "strength": q.get("strength", 0.5),

        "display_tags": "|".join(map(str, track.get("display_tags", []))),
        "va": json.dumps(track.get("va", []), ensure_ascii=False),
        "audio_path": audio_path,
    }


def write_csv(rows):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "method", "query_id", "rank", "song_id", "score",
        "text", "current_state", "regulation_goal",
        "strategy", "strategy_zh", "strength",
        "display_tags", "va", "audio_path",
    ]

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    queries = load_queries()
    query_ids = [qid(q) for q in queries]
    query_texts = [query_text(q) for q in queries]

    audio = load_audio_emb()
    text_emb = load_or_build_text_emb(query_ids, query_texts)
    tracks = load_track_meta()

    scores = norm(audio["audio_emb"]) @ norm(text_emb).T
    rows = []

    for qi, q in enumerate(queries):
        top_idx = np.argsort(-scores[:, qi])[:TOP_K]

        for rank, idx in enumerate(top_idx, start=1):
            idx = int(idx)
            song_id = audio["song_ids"][idx]
            track = tracks.get(song_id, {})
            rows.append(
                make_row(
                    q=q,
                    song_id=song_id,
                    audio_path=audio["audio_paths"][idx],
                    score=scores[idx, qi],
                    rank=rank,
                    track=track,
                )
            )

    write_csv(rows)

    print(f"完成 CLAP Top-{TOP_K}: {OUT_CSV}")
    print(f"输出行数: {len(rows)}")


if __name__ == "__main__":
    main()