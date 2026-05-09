from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from core.config import music_db_path
from core.utils import load_music_db


def _to_float_list(values: Sequence[object]) -> List[float]:
    return [float(v) for v in values]


def normalize_track(raw: Dict[str, object]) -> Optional[Dict[str, object]]:
    vec28 = raw.get("semantic_vector_28d")
    try:
        vector = _to_float_list(vec28[:28])
    except (TypeError, ValueError):
        return None

    return {
        "song_id": str(raw.get("song_id", "Unknown")),
        "song_path": str(raw.get("song_path", "")),
        "display_tags": list(raw.get("display_tags", [])) if isinstance(raw.get("display_tags", []), list) else [],
        "va": [vector[0], vector[1]],
        "tag26": vector[2:28],
        "semantic_vector_28d": vector,
        "raw": raw,
    }


def load_tracks(db_path: Optional[Path] = None) -> List[Dict[str, object]]:
    tracks: List[Dict[str, object]] = []
    for item in load_music_db(db_path or music_db_path):
        if isinstance(item, dict):
            track = normalize_track(item)
            if track is not None:
                tracks.append(track)
    return tracks


if __name__ == "__main__":
    tracks = load_tracks()
    print(f"Loaded {len(tracks)} tracks")
    if tracks:
        print("Example track:", tracks[0])
