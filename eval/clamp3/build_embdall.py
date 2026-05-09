# -*- coding: utf-8 -*-

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from core.music_repository import load_tracks


# =========================
# Paths
# =========================

OUT_ROOT = Path(r"D:\AAAA_audio_dataset\eval\clamp3")

# 旧的 1000 首结果
OLD_MERT_DIR = OUT_ROOT / "mert"

# 全量结果
MERT_DIR = OUT_ROOT / "mert_all"
EMBD_DIR = OUT_ROOT / "audio_embd_all"

# 全量临时音频目录
TMP_AUDIO_DIR = OUT_ROOT / "_audio_tmp_all"

# CLaMP3 官方目录
CLAMP3_ROOT = ROOT / "tools" / "basline" / "clamp3"
EXTRACT_MERT = CLAMP3_ROOT / "preprocessing" / "audio" / "extract_mert.py"
CLAMP3_EMBD = CLAMP3_ROOT / "clamp3_embd.py"

MODEL_PATH = "m-a-p/MERT-v1-95M"

# 每批处理数量。
# 1000 比较稳，太大可能临时目录太多文件，太小会频繁重新加载模型。
BATCH_SIZE = 1000


# =========================
# Utils
# =========================

def safe_stem(song_id: str) -> str:
    """
    把 song_id 转成 Windows 安全文件名。
    正常 song_id 应该是 track_123456，这里只是防止特殊字符。
    """
    return re.sub(r'[<>:"/\\\\|?*]+', "_", str(song_id))


def mert_path_for(song_id: str) -> Path:
    return MERT_DIR / f"{safe_stem(song_id)}.npy"


def mert_done(song_id: str) -> bool:
    p = mert_path_for(song_id)
    return p.exists() and p.stat().st_size > 0


def count_npy(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.rglob("*.npy"))


def clear_tmp_audio_dir():
    """
    每个 batch 开始前清空临时目录。
    这样 _audio_tmp_all 不会一直累积，也不会重复处理旧批次。
    """
    if TMP_AUDIO_DIR.exists():
        shutil.rmtree(TMP_AUDIO_DIR)

    TMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def link_or_copy_audio(src: Path, dst: Path):
    """
    不移动原始音频。

    优先硬链接：
    - 不额外占用一份音频空间
    - 原始文件不移动

    如果硬链接失败：
    - 用 copy2 复制一份到临时目录
    - 原始文件仍然不动
    """
    if dst.exists():
        return

    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


# =========================
# Data collection
# =========================

def collect_audio_items():
    """
    从项目已有 load_tracks() 读取全部音乐。
    按 song_id 去重。
    """
    tracks = load_tracks()

    items = []
    seen = set()

    for t in tracks:
        song_id = str(t.get("song_id", "")).strip()
        audio_path = Path(str(t.get("song_path", "")).strip())

        if not song_id:
            continue

        if song_id in seen:
            continue

        if not audio_path.exists():
            continue

        seen.add(song_id)
        items.append((song_id, audio_path))

    return items


def copy_old_mert_if_exists():
    """
    把旧的 1000 首 MERT 复制到 mert_all。
    如果已经复制过，就跳过。
    """
    if not OLD_MERT_DIR.exists():
        print(f"旧 MERT 目录不存在，跳过复用: {OLD_MERT_DIR}")
        return

    MERT_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for p in OLD_MERT_DIR.rglob("*.npy"):
        dst = MERT_DIR / p.name

        if dst.exists() and dst.stat().st_size > 0:
            skipped += 1
            continue

        try:
            shutil.copy2(p, dst)
            copied += 1
        except Exception as e:
            print(f"[复制旧 MERT 失败] {p}")
            print(repr(e))

    print(f"旧 MERT 复用完成: copied={copied}, skipped={skipped}")


# =========================
# MERT extraction
# =========================

def prepare_batch_audio(batch):
    """
    准备当前 batch 的临时音频。
    CLaMP3 的 extract_mert.py 是按文件夹处理的，
    所以每次把一批音频放到 _audio_tmp_all。
    """
    clear_tmp_audio_dir()

    prepared = 0

    for song_id, audio_path in batch:
        stem = safe_stem(song_id)
        suffix = audio_path.suffix if audio_path.suffix else ".mp3"

        dst = TMP_AUDIO_DIR / f"{stem}{suffix}"

        try:
            link_or_copy_audio(audio_path, dst)
            prepared += 1
        except Exception as e:
            print(f"[临时音频失败] {song_id} {audio_path}")
            print(repr(e))

    return prepared


def run_extract_mert_on_tmp():
    if not EXTRACT_MERT.exists():
        raise FileNotFoundError(f"找不到 extract_mert.py: {EXTRACT_MERT}")

    MERT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(EXTRACT_MERT),
        "--input_path",
        str(TMP_AUDIO_DIR),
        "--output_path",
        str(MERT_DIR),
        "--model_path",
        MODEL_PATH,
        "--mean_features",
    ]

    print("开始生成本批 MERT:")
    print(" ".join(cmd))

    subprocess.run(cmd, cwd=str(CLAMP3_ROOT), check=True)


def build_all_mert():
    MERT_DIR.mkdir(parents=True, exist_ok=True)

    copy_old_mert_if_exists()

    items = collect_audio_items()

    print("=" * 80)
    print(f"全部音乐数量: {len(items)}")
    print(f"MERT 输出目录: {MERT_DIR}")
    print(f"当前 MERT 文件数量: {count_npy(MERT_DIR)}")

    pending = [
        (song_id, audio_path)
        for song_id, audio_path in items
        if not mert_done(song_id)
    ]

    print(f"已有 MERT 数量: {len(items) - len(pending)}")
    print(f"待生成 MERT 数量: {len(pending)}")

    if not pending:
        print("MERT 已全部完成。")
        return

    total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(pending))
        batch = pending[start:end]

        print("=" * 80)
        print(f"MERT 批次 {batch_idx + 1}/{total_batches}")
        print(f"本批范围: {start + 1}-{end} / {len(pending)}")
        print(f"本批数量: {len(batch)}")

        prepared = prepare_batch_audio(batch)

        print(f"临时音频准备数量: {prepared}")
        print(f"临时目录: {TMP_AUDIO_DIR}")

        if prepared == 0:
            print("本批没有可用音频，跳过。")
            continue

        run_extract_mert_on_tmp()

        done_now = sum(1 for song_id, _ in items if mert_done(song_id))

        print(f"当前 MERT 完成数量: {done_now}/{len(items)}")
        print(f"当前 MERT 文件数量: {count_npy(MERT_DIR)}")

    if TMP_AUDIO_DIR.exists():
        shutil.rmtree(TMP_AUDIO_DIR)
        print(f"已清理临时目录: {TMP_AUDIO_DIR}")

    print("=" * 80)
    print(f"完成 MERT: {MERT_DIR}")
    print(f"MERT 文件数量: {count_npy(MERT_DIR)}")


# =========================
# CLaMP3 embedding
# =========================

def build_clamp3_embd():
    if not CLAMP3_EMBD.exists():
        raise FileNotFoundError(f"找不到 clamp3_embd.py: {CLAMP3_EMBD}")

    if not MERT_DIR.exists():
        raise FileNotFoundError(f"找不到 MERT 目录: {MERT_DIR}")

    EMBD_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(CLAMP3_EMBD),
        str(MERT_DIR),
        str(EMBD_DIR),
        "--get_global",
    ]

    print("=" * 80)
    print("开始生成 CLaMP3 audio embedding:")
    print(" ".join(cmd))

    subprocess.run(cmd, cwd=str(CLAMP3_ROOT), check=True)

    print("=" * 80)
    print(f"完成 CLaMP3 embedding: {EMBD_DIR}")
    print(f"embedding 文件数量: {count_npy(EMBD_DIR)}")


# =========================
# Main
# =========================

def main():
    print("CLaMP3 全量构建开始")
    print(f"项目根目录: {ROOT}")
    print(f"CLaMP3 根目录: {CLAMP3_ROOT}")
    print(f"旧 MERT 目录: {OLD_MERT_DIR}")
    print(f"全量 MERT 目录: {MERT_DIR}")
    print(f"全量 embedding 目录: {EMBD_DIR}")
    print(f"临时音频目录: {TMP_AUDIO_DIR}")

    build_all_mert()
    build_clamp3_embd()

    print("CLaMP3 全量构建完成")


if __name__ == "__main__":
    main()