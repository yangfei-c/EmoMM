import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")
INPUT_DB = ROOT / "data" / "music_db" / "semantic_db.json"
OUTPUT_DIR = ROOT / "src" / "eval_experiment" / "data"
OUTPUT_DB = OUTPUT_DIR / "music_eval_db.json"

def normalize_one_song(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    vec = item["semantic_vector_28d"]
    music_va = vec[:2]
    tag_vector_26d = vec[2:]
    return {
        "song_id": item["song_id"],
        "source": item.get("source", "unknown"),
        "audio_path": item["song_path"],
        "display_tags": item.get("display_tags", []),
        "music_va": music_va,
        "tag_vector_26d": tag_vector_26d,
    }

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"读取音乐库: {INPUT_DB}")
    with INPUT_DB.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    normalized: List[Dict[str, Any]] = []

    for idx, item in enumerate(raw_data):
        normalized.append(normalize_one_song(item, idx))

    with OUTPUT_DB.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    print("处理完成！")

if __name__ == "__main__":
    main()
