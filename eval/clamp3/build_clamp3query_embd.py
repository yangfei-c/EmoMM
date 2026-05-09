# -*- coding: utf-8 -*-

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 注意：这里用你当前项目里的文件名
QUERY_JSON = ROOT / "data" / "query" / "queries_zh16.json"

CLAMP3_DIR = ROOT / "tools" / "basline" / "clamp3"
CLAMP3_EMBD = CLAMP3_DIR / "clamp3_embd.py"

# 大文件全部放到项目外，避免 PyCharm 卡顿
CLAMP3_EVAL_DIR = Path(r"D:\AAAA_audio_dataset\eval")

QUERY_TXT_DIR = CLAMP3_EVAL_DIR / "query_txt"
QUERY_EMB_DIR = CLAMP3_EVAL_DIR / "query"
QUERY_TEXT_JSONL = CLAMP3_EVAL_DIR / "query_texts.jsonl"


def main():
    if not QUERY_JSON.exists():
        raise FileNotFoundError(f"找不到 query 文件：{QUERY_JSON}")

    if not CLAMP3_EMBD.exists():
        raise FileNotFoundError(f"找不到 clamp3_embd.py：{CLAMP3_EMBD}")

    queries = json.loads(QUERY_JSON.read_text(encoding="utf-8"))

    CLAMP3_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if QUERY_TXT_DIR.exists():
        shutil.rmtree(QUERY_TXT_DIR)
    QUERY_TXT_DIR.mkdir(parents=True, exist_ok=True)

    # 同时生成 txt 文件夹和 jsonl 记录
    with QUERY_TEXT_JSONL.open("w", encoding="utf-8") as jf:
        for q in queries:
            qid = q["query_id"]
            text = q["text"]

            (QUERY_TXT_DIR / f"{qid}.txt").write_text(text, encoding="utf-8")
            jf.write(json.dumps({"id": qid, "text": text}, ensure_ascii=False) + "\n")

    if QUERY_EMB_DIR.exists():
        shutil.rmtree(QUERY_EMB_DIR)

    cmd = [
        sys.executable,
        str(CLAMP3_EMBD),
        str(QUERY_TXT_DIR),
        str(QUERY_EMB_DIR),
        "--get_global",
    ]

    print("开始生成 CLaMP3 query embedding...")
    print("命令：")
    print(" ".join(cmd))

    subprocess.run(cmd, cwd=str(CLAMP3_DIR), check=True)

    print("\n完成。")
    print(f"query txt 目录：{QUERY_TXT_DIR}")
    print(f"query jsonl 文件：{QUERY_TEXT_JSONL}")
    print(f"query embedding 目录：{QUERY_EMB_DIR}")


if __name__ == "__main__":
    main()