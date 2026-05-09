import json
from pathlib import Path

ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

QUERY_JSON = ROOT /  "eval" /  "queries_zh16.json"
OUT_DIR = Path(r"D:\AAAA_audio_dataset\eval\clamp3\query_txt")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    queries = json.loads(QUERY_JSON.read_text(encoding="utf-8"))

    for q in queries:
        qid = str(q.get("query_id") or q.get("id"))
        text = q["text"]
        (OUT_DIR / f"{qid}.txt").write_text(text, encoding="utf-8")

    print(f"完成: {OUT_DIR}")
    print(f"query 数量: {len(queries)}")


if __name__ == "__main__":
    main()