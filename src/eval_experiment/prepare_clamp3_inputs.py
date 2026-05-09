# -*- coding: utf-8 -*-
"""
prepare_clamp3_inputs.py

作用：
1. 读取 query_test_32.json
2. 为每条 query 生成一个 CLaMP3 文本 prompt 文件：.txt
3. 生成 clamp3_query_manifest.csv，记录 query_id 与 prompt 文件路径

输出：
src/eval_experiment/clamp3_inputs/text_queries/*.txt
src/eval_experiment/clamp3_inputs/clamp3_query_manifest.csv

运行：
cd /d D:\AAAAAAAAAA_emotion\Code\EmoMusicRec
python src\eval_experiment\prepare_clamp3_inputs.py
"""

import csv
import json
from pathlib import Path


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

QUERY_PATH = ROOT / "src" / "eval_experiment" / "data" / "query_test_32.json"

CLAMP3_INPUT_DIR = ROOT / "src" / "eval_experiment" / "clamp3_inputs"
TEXT_QUERY_DIR = CLAMP3_INPUT_DIR / "text_queries"
MANIFEST_PATH = CLAMP3_INPUT_DIR / "clamp3_query_manifest.csv"


def safe_text(s: str) -> str:
    return s.replace("\n", " ").replace("\r", " ").strip()


def main() -> None:
    TEXT_QUERY_DIR.mkdir(parents=True, exist_ok=True)

    with QUERY_PATH.open("r", encoding="utf-8") as f:
        queries = json.load(f)

    rows = []

    for q in queries:
        query_id = q["query_id"]

        # 优先使用已经生成的英文 prompt；没有则退回中文 prompt；再没有就用 text + strategy
        prompt = q.get("prompt_en") or q.get("prompt_cn") or f"{q.get('text', '')} {q.get('strategy', '')}"
        prompt = safe_text(prompt)

        prompt_file = TEXT_QUERY_DIR / f"{query_id}.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        rows.append({
            "query_id": query_id,
            "strategy": q.get("strategy", ""),
            "language": q.get("language", ""),
            "prompt_file": str(prompt_file),
            "prompt": prompt,
        })

    with MANIFEST_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "strategy", "language", "prompt_file", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print("CLaMP3 text query files generated.")
    print(f"text query dir: {TEXT_QUERY_DIR}")
    print(f"manifest: {MANIFEST_PATH}")
    print(f"num queries: {len(rows)}")


if __name__ == "__main__":
    main()
