# -*- coding: utf-8 -*-

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")
QUERY_JSON = ROOT / "eval" / "queries_zh16.json"

BASE_DIR = Path(r"D:\AAAA_audio_dataset\eval\clamp3")
TXT_DIR = BASE_DIR / "query_txt"
EMBD_DIR = BASE_DIR / "query_embd"

CLAMP3_CODE_DIR = ROOT / "tools" / "basline" / "clamp3" / "code"
EXTRACT_SCRIPT = CLAMP3_CODE_DIR / "extract_clamp3.py"

CLEAR_OLD = True


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


def reset_dir(path):
    if CLEAR_OLD and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_query_txt(queries):
    reset_dir(TXT_DIR)

    for q in queries:
        (TXT_DIR / f"{qid(q)}.txt").write_text(query_text(q), encoding="utf-8")

    print(f"完成 query txt: {TXT_DIR}")
    print(f"txt 数量: {len(list(TXT_DIR.glob('*.txt')))}")


def build_embedding():
    reset_dir(EMBD_DIR)

    if not EXTRACT_SCRIPT.exists():
        raise FileNotFoundError(EXTRACT_SCRIPT)

    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "extract_clamp3.py",
        str(TXT_DIR),
        str(EMBD_DIR),
        "--get_global",
    ]

    print("开始生成 CLaMP3 query embedding:")
    print(" ".join(cmd))

    subprocess.run(cmd, cwd=str(CLAMP3_CODE_DIR), check=True)

    print(f"完成 query embedding: {EMBD_DIR}")
    print(f"embedding 数量: {len(list(EMBD_DIR.glob('*.npy')))}")


def main():
    queries = load_queries()

    write_query_txt(queries)
    build_embedding()

    expected = len(queries)
    actual = len(list(EMBD_DIR.glob("*.npy")))

    if actual != expected:
        raise RuntimeError(f"query embedding 数量不对: {actual} != {expected}")

    print("CLaMP3 query embedding 生成完成。")


if __name__ == "__main__":
    main()