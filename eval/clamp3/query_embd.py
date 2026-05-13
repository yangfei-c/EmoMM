# -*- coding: utf-8 -*-

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

QUERY_JSON = ROOT / "eval" / "queries_zh16.json"

BASE_DIR = Path(r"D:\AAAA_audio_dataset\eval\clamp3")
OUT_TXT_DIR = BASE_DIR / "query_txt"
OUT_EMBD_DIR = BASE_DIR / "query_embd"

CLAMP3_ROOT = ROOT / "tools" / "basline" / "clamp3"
CLAMP3_CODE_DIR = CLAMP3_ROOT / "code"
EXTRACT_CLAMP3 = CLAMP3_CODE_DIR / "extract_clamp3.py"

# 正式实验建议每次 query 改完都清空旧 query embedding，避免误用旧结果
CLEAR_OLD_QUERY_TXT = True
CLEAR_OLD_QUERY_EMBD = True


def get_query_id(q):
    qid = q.get("query_id") or q.get("id")
    if not qid:
        raise ValueError(f"query 缺少 query_id/id: {q}")
    return str(qid)


def get_retrieval_text(q):
    """
    CLaMP3 baseline 使用完整查询文本：
    current_state + regulation_goal。

    新版 JSON 推荐有 text 字段；
    如果没有 text，则自动拼 current_state 和 regulation_goal。
    """
    text = str(q.get("text") or "").strip()

    if text:
        return text

    current_state = str(q.get("current_state") or "").strip()
    regulation_goal = str(q.get("regulation_goal") or "").strip()
    text = f"{current_state}{regulation_goal}".strip()

    if not text:
        raise ValueError(f"{get_query_id(q)} 缺少 text/current_state/regulation_goal")

    return text


def check_required_files():
    if not QUERY_JSON.exists():
        raise FileNotFoundError(f"找不到 query JSON: {QUERY_JSON}")

    if not CLAMP3_ROOT.exists():
        raise FileNotFoundError(f"找不到 CLaMP3 根目录: {CLAMP3_ROOT}")

    if not CLAMP3_CODE_DIR.exists():
        raise FileNotFoundError(f"找不到 CLaMP3 code 目录: {CLAMP3_CODE_DIR}")

    if not EXTRACT_CLAMP3.exists():
        raise FileNotFoundError(f"找不到 extract_clamp3.py: {EXTRACT_CLAMP3}")


def clear_folder(folder: Path):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def write_query_txt_files():
    if CLEAR_OLD_QUERY_TXT:
        clear_folder(OUT_TXT_DIR)
    else:
        OUT_TXT_DIR.mkdir(parents=True, exist_ok=True)

    queries = json.loads(QUERY_JSON.read_text(encoding="utf-8"))

    if not isinstance(queries, list):
        raise ValueError("queries_zh16.json 顶层必须是 list")

    written = []

    for q in queries:
        qid = get_query_id(q)
        text = get_retrieval_text(q)

        out_path = OUT_TXT_DIR / f"{qid}.txt"
        out_path.write_text(text, encoding="utf-8")

        written.append((qid, text, out_path))

    print("=" * 80)
    print(f"完成 query txt 生成: {OUT_TXT_DIR}")
    print(f"query 数量: {len(written)}")

    for qid, text, out_path in written[:5]:
        print(f"{qid}: {text}")
        print(f"  -> {out_path}")

    return written


def build_query_embedding():
    if CLEAR_OLD_QUERY_EMBD:
        clear_folder(OUT_EMBD_DIR)
    else:
        OUT_EMBD_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "extract_clamp3.py",
        str(OUT_TXT_DIR),
        str(OUT_EMBD_DIR),
        "--get_global",
    ]

    print("=" * 80)
    print("开始生成 CLaMP3 query text embedding:")
    print(" ".join(cmd))
    print(f"工作目录: {CLAMP3_CODE_DIR}")

    subprocess.run(
        cmd,
        cwd=str(CLAMP3_CODE_DIR),
        check=True,
    )

    print("=" * 80)
    print(f"完成 CLaMP3 query embedding: {OUT_EMBD_DIR}")
    print(f"embedding 文件数量: {count_npy(OUT_EMBD_DIR)}")


def count_npy(folder: Path):
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.glob("*.npy"))


def validate_outputs(expected_query_count):
    txt_count = len(list(OUT_TXT_DIR.glob("*.txt")))
    embd_count = len(list(OUT_EMBD_DIR.glob("*.npy")))

    print("=" * 80)
    print("输出检查")
    print(f"txt 数量: {txt_count}")
    print(f"embedding 数量: {embd_count}")
    print(f"期望 query 数量: {expected_query_count}")

    if txt_count != expected_query_count:
        raise RuntimeError(f"query txt 数量不对: {txt_count} != {expected_query_count}")

    if embd_count != expected_query_count:
        raise RuntimeError(f"query embedding 数量不对: {embd_count} != {expected_query_count}")

    print("CLaMP3 query text embedding 生成检查通过。")


def main():
    print("CLaMP3 query text embedding 构建开始")
    print(f"项目根目录: {ROOT}")
    print(f"query JSON: {QUERY_JSON}")
    print(f"query txt 输出目录: {OUT_TXT_DIR}")
    print(f"query embedding 输出目录: {OUT_EMBD_DIR}")
    print(f"CLaMP3 code 目录: {CLAMP3_CODE_DIR}")

    check_required_files()

    written = write_query_txt_files()
    build_query_embedding()
    validate_outputs(expected_query_count=len(written))

    print("=" * 80)
    print("CLaMP3 query text embedding 构建完成")


if __name__ == "__main__":
    main()