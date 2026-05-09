# -*- coding: utf-8 -*-
"""
从已有 MERT 特征生成 CLaMP3 全局检索 embedding。

输入：
D:\AAAA_audio_dataset\eval\mert

中间目录：
D:\AAAA_audio_dataset\eval\mert_flat_for_clamp3

输出：
D:\AAAA_audio_dataset\eval\audio_embd

运行：
cd /d D:\AAAAAAAAAA_emotion\Code\EmoMusicRec
D:\Anaconda3\envs\music\python.exe tools\analysis\build_clamp3_audio_embd_from_mert.py
"""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

CLAMP3_DIR = ROOT / "tools" / "basline" / "clamp3"
CLAMP3_EMBD = CLAMP3_DIR / "clamp3_embd.py"

# 原始 MERT 特征，允许有子目录
MERT_INPUT_DIR = Path(r"D:\AAAA_audio_dataset\eval\mert")

# 扁平化后的 MERT 输入目录，给 clamp3_embd.py 用
MERT_FLAT_DIR = Path(r"D:\AAAA_audio_dataset\eval\mert_flat_for_clamp3")

# CLaMP3 最终检索 embedding 输出目录
AUDIO_EMBD_DIR = Path(r"D:\AAAA_audio_dataset\eval\audio_embd")

# 记录原始路径和新文件名的映射
MAPPING_CSV = Path(r"D:\AAAA_audio_dataset\eval\mert_flat_mapping.csv")


def safe_load_shape(path: Path):
    """
    只检查 .npy 是否可读，并返回 shape。
    mmap_mode='r' 尽量避免一次性读入大数组。
    """
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return arr.shape, None
    except Exception as e:
        return None, repr(e)


def make_unique_id(path: Path, used_ids: set) -> str:
    """
    优先用文件名作为 id。
    如果重复，则用父目录名 + 文件名。
    """
    base = path.stem

    if base not in used_ids:
        used_ids.add(base)
        return base

    base2 = f"{path.parent.name}__{path.stem}"
    if base2 not in used_ids:
        used_ids.add(base2)
        return base2

    i = 1
    while True:
        candidate = f"{path.parent.name}__{path.stem}__{i}"
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate
        i += 1


def prepare_flat_mert_dir():
    if not MERT_INPUT_DIR.exists():
        raise FileNotFoundError(f"找不到 MERT 输入目录：{MERT_INPUT_DIR}")

    npy_files = sorted(MERT_INPUT_DIR.rglob("*.npy"))
    if not npy_files:
        raise RuntimeError(f"MERT 输入目录里没有 .npy：{MERT_INPUT_DIR}")

    if MERT_FLAT_DIR.exists():
        shutil.rmtree(MERT_FLAT_DIR)
    MERT_FLAT_DIR.mkdir(parents=True, exist_ok=True)

    MAPPING_CSV.parent.mkdir(parents=True, exist_ok=True)

    used_ids = set()
    ok_count = 0
    bad_count = 0

    with MAPPING_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "new_id",
                "new_file",
                "source_file",
                "shape",
                "status",
                "error",
            ],
        )
        writer.writeheader()

        for idx, src in enumerate(npy_files, start=1):
            shape, error = safe_load_shape(src)

            if error is not None:
                bad_count += 1
                writer.writerow({
                    "new_id": "",
                    "new_file": "",
                    "source_file": str(src),
                    "shape": "",
                    "status": "bad",
                    "error": error,
                })

                if bad_count <= 20:
                    print(f"[BAD] {src} | {error}")
                continue

            new_id = make_unique_id(src, used_ids)
            dst = MERT_FLAT_DIR / f"{new_id}.npy"

            shutil.copy2(src, dst)
            ok_count += 1

            writer.writerow({
                "new_id": new_id,
                "new_file": str(dst),
                "source_file": str(src),
                "shape": str(shape),
                "status": "ok",
                "error": "",
            })

            if ok_count % 500 == 0:
                print(f"已复制可用 MERT 文件：{ok_count}")

    print("\nMERT 扁平化完成")
    print(f"原始 .npy 总数：{len(npy_files)}")
    print(f"可用 .npy 数量：{ok_count}")
    print(f"损坏/空文件数量：{bad_count}")
    print(f"扁平化目录：{MERT_FLAT_DIR}")
    print(f"映射表：{MAPPING_CSV}")

    if ok_count == 0:
        raise RuntimeError("没有任何可用 MERT 文件，不能继续生成 CLaMP3 embedding。")


def run_clamp3_embd():
    if not CLAMP3_EMBD.exists():
        raise FileNotFoundError(f"找不到 clamp3_embd.py：{CLAMP3_EMBD}")

    if AUDIO_EMBD_DIR.exists():
        print(f"\n删除旧输出目录：{AUDIO_EMBD_DIR}")
        shutil.rmtree(AUDIO_EMBD_DIR)

    cmd = [
        sys.executable,
        str(CLAMP3_EMBD),
        str(MERT_FLAT_DIR),
        str(AUDIO_EMBD_DIR),
        "--get_global",
    ]

    print("\n开始调用 CLaMP3 生成 audio embedding...")
    print("命令：")
    print(" ".join(cmd))

    subprocess.run(cmd, cwd=str(CLAMP3_DIR), check=True)

    out_files = list(AUDIO_EMBD_DIR.rglob("*.npy"))
    print("\nCLaMP3 audio embedding 生成完成")
    print(f"输出目录：{AUDIO_EMBD_DIR}")
    print(f"输出 .npy 数量：{len(out_files)}")

    if out_files:
        sample = out_files[0]
        arr = np.load(sample, allow_pickle=False)
        print(f"示例输出：{sample}")
        print(f"示例 shape：{arr.shape}")

    if not out_files:
        raise RuntimeError("CLaMP3 输出目录中没有 .npy 文件，生成失败。")


def main():
    print(f"项目根目录：{ROOT}")
    print(f"MERT 输入目录：{MERT_INPUT_DIR}")

    prepare_flat_mert_dir()
    run_clamp3_embd()

    print("\n下一步请修改 clamp3_top5.py：")
    print(f'MUSIC_EMB_DIR = Path(r"{AUDIO_EMBD_DIR}")')
    print('QUERY_EMB_DIR = Path(r"D:\\AAAA_audio_dataset\\eval\\query")')


if __name__ == "__main__":
    main()