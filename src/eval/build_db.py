from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JAMENDO_JSON = PROJECT_ROOT / "data" / "MTG-Jamendo" / "jamendo.json"
TAGS_26_NPY = PROJECT_ROOT / "data" / "MTG-Jamendo" / "tags_26.npy"
MERT_ROOT = Path(r"D:\AAAA_audio_dataset\mtg-jamendo\MERT_95M_30s")
AUDIO_ROOT = Path(r"D:\AAAA_audio_dataset\mtg-jamendo\audio")
TAG_CKPT = PROJECT_ROOT / "src" / "eval" / "tag.ckpt"
MUSIC_VA_CKPT = PROJECT_ROOT / "src" / "eval" / "m_va.ckpt"
OUT_JSON = PROJECT_ROOT / "data" / "music_db" / "semantic_db.json"
OUT_PT = PROJECT_ROOT / "data" / "music_db" / "semantic_db.pt"


def setup_imports() -> None:
    src = PROJECT_ROOT / "src"
    for p in [src, src / "music_tag", src / "music_va"]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def strip_tag_name(tag: str) -> str:
    return str(tag).split("---")[-1]


def load_mert(song_dir: Path) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not song_dir.exists() or not song_dir.is_dir():
        return None, None

    feats: List[torch.Tensor] = []
    for p in sorted(song_dir.glob("*.npy")):
        try:
            t = torch.from_numpy(np.load(str(p))).float()
            while t.dim() > 1:
                t = t.mean(dim=0)
            if t.numel() == 768:
                feats.append(t)
        except Exception:
            continue

    if not feats:
        return None, None

    stack = torch.stack(feats, dim=0)
    feat_768 = stack.mean(dim=0)
    feat_1536 = torch.cat([feat_768, stack.max(dim=0).values], dim=-1)
    return feat_768.unsqueeze(0), feat_1536.unsqueeze(0)


def resolve_audio_path(rel_path: str) -> str:
    rel = Path(rel_path.replace("/", "\\"))
    base = AUDIO_ROOT / rel
    if base.exists():
        return str(base)
    for ext in [".mp3", ".ogg", ".wav", ".flac"]:
        c = base.with_suffix(ext)
        if c.exists():
            return str(c)
    return str(base.with_suffix(".mp3"))


def top1_from_probs(probs: np.ndarray, names: np.ndarray) -> str:
    return strip_tag_name(str(names[int(np.argmax(probs))]))


def build_db() -> None:
    setup_imports()
    from music_tag.trainer import MusicTagTrainer
    from music_va.trainer import MusicVATrainer

    for p in [JAMENDO_JSON, TAGS_26_NPY, TAG_CKPT, MUSIC_VA_CKPT]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_PT.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    tags = np.load(str(TAGS_26_NPY), allow_pickle=True)
    genre_names, inst_names, mood_names = tags[:10], tags[10:17], tags[17:]

    tag_model = MusicTagTrainer.load_from_checkpoint(str(TAG_CKPT), map_location=device).to(device).eval()
    va_model = MusicVATrainer.load_from_checkpoint(str(MUSIC_VA_CKPT), map_location=device).to(device).eval()

    with JAMENDO_JSON.open("r", encoding="utf-8") as f:
        items = json.load(f)

    rows: List[Dict] = []
    song_ids: List[str] = []
    song_paths: List[str] = []
    display_tags_all: List[List[str]] = []
    vecs_28: List[List[float]] = []
    va_2d_all: List[List[float]] = []
    tag_probs_all: List[List[float]] = []

    for item in tqdm(items, desc="Build DB"):
        rel_path = str(item.get("path", "")).replace("/", "\\")
        if not rel_path:
            continue

        feat_768, feat_1536 = load_mert(MERT_ROOT / rel_path)
        if feat_768 is None or feat_1536 is None:
            continue

        with torch.no_grad():
            va = va_model(feat_1536.to(device))[0].detach().cpu().numpy().astype(np.float64)
            out = tag_model(feat_768.to(device))
            genre = torch.sigmoid(out["genre"])[0].detach().cpu().numpy().astype(np.float64)
            inst = torch.sigmoid(out["instrument"])[0].detach().cpu().numpy().astype(np.float64)
            mood = torch.sigmoid(out["mood"])[0].detach().cpu().numpy().astype(np.float64)

        probs_26 = np.concatenate([genre, inst, mood], axis=0)
        display_tags = [
            top1_from_probs(genre, genre_names),
            top1_from_probs(inst, inst_names),
            top1_from_probs(mood, mood_names),
        ]
        vec28 = np.concatenate([va, probs_26], axis=0).tolist()

        song_id = str(item.get("track_id", item.get("id", "")))
        song_path = resolve_audio_path(rel_path)

        rows.append(
            {
                "song_id": song_id,
                "source": "jamendo",
                "song_path": song_path,
                "display_tags": display_tags,
                "semantic_vector_28d": vec28,
            }
        )
        song_ids.append(song_id)
        song_paths.append(song_path)
        display_tags_all.append(display_tags)
        vecs_28.append(vec28)
        va_2d_all.append(va.tolist())
        tag_probs_all.append(probs_26.tolist())

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    torch.save(
        {
            "song_ids": song_ids,
            "song_paths": song_paths,
            "display_tags": display_tags_all,
            "semantic_vector_28d": torch.tensor(vecs_28, dtype=torch.float32),
            "va_2d": torch.tensor(va_2d_all, dtype=torch.float32),
            "tag_probs_26": torch.tensor(tag_probs_all, dtype=torch.float32),
        },
        str(OUT_PT),
    )

    print(f"\n[OK] Saved JSON: {OUT_JSON}")
    print(f"[OK] Saved PT:   {OUT_PT}")
    print(f"[OK] Total songs: {len(rows)}")


if __name__ == "__main__":
    build_db()
