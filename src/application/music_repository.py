from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from application.config import music_db_path, tags_26
from application.utils import load_music_db


def _to_float_list(values: Sequence[object]) -> List[float]:
    return [float(v) for v in values]


def normalize_track(raw: Dict[str, object]) -> Optional[Dict[str, object]]:
    vec28 = raw.get("semantic_vector_28d")
    if not isinstance(vec28, list) or len(vec28) < 28:
        return None
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


def track_by_id(tracks: Sequence[Dict[str, object]], song_id: str) -> Optional[Dict[str, object]]:
    return next((t for t in tracks if str(t.get("song_id", "")) == str(song_id)), None)


def as_arrays(tracks: Sequence[Dict[str, object]]) -> Tuple[List[str], List[List[float]], List[List[float]]]:
    song_ids: List[str] = []
    va_matrix: List[List[float]] = []
    tag26_matrix: List[List[float]] = []

    for track in tracks:
        tag26 = [float(x) for x in track.get("tag26", [])]
        if len(tag26) != len(tags_26):
            continue
        song_ids.append(str(track.get("song_id", "Unknown")))
        va_matrix.append([float(x) for x in track.get("va", [0.0, 0.0])])
        tag26_matrix.append(tag26)

    return song_ids, va_matrix, tag26_matrix


if __name__ == "__main__":
    tracks = load_tracks()
    print(f"Loaded {len(tracks)} tracks")
    if tracks:
        print("Example track:", tracks[0])
