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

# 你的截图目录：
# D:\AAAA_audio_dataset\eval\clamp3
OUT_ROOT = Path(r"D:\AAAA_audio_dataset\eval\clamp3")

# 已经生成好的 MERT 特征目录
# 里面应该是 track_xxxxxx.npy
MERT_DIR = OUT_ROOT / "mert"

# 准备输出 CLaMP3 audio embedding 的目录
EMBD_DIR = OUT_ROOT / "audio_embd"

# 如果后续还有缺失 MERT，需要临时放音频
TMP_AUDIO_DIR = OUT_ROOT / "_audio_tmp"

# CLaMP3 项目目录
CLAMP3_ROOT = ROOT / "tools" / "basline" / "clamp3"

# MERT 提取脚本：只在发现 MERT 不完整时才会用
EXTRACT_MERT = CLAMP3_ROOT / "preprocessing" / "audio" / "extract_mert.py"

# 关键修改：不要再调用 clamp3_embd.py
# 已有 MERT .npy 后，应该直接调用 code/extract_clamp3.py
CLAMP3_CODE_DIR = CLAMP3_ROOT / "code"
EXTRACT_CLAMP3 = CLAMP3_CODE_DIR / "extract_clamp3.py"

MODEL_PATH = "m-a-p/MERT-v1-95M"

# 每批处理数量；只有补 MERT 时才用
BATCH_SIZE = 1000

# 你现在 MERT 已经全好了，所以默认不强制重跑 MERT
RUN_MERT_IF_MISSING = True

# 如果 audio_embd 已经和 mert 数量一致，是否跳过 CLaMP3 embedding
SKIP_EMBD_IF_COMPLETE = True


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
    if TMP_AUDIO_DIR.exists():
        shutil.rmtree(TMP_AUDIO_DIR)

    TMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def link_or_copy_audio(src: Path, dst: Path):
    """
    不移动原始音频。
    优先硬链接；失败则复制。
    """
    if dst.exists():
        return

    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def check_required_files():
    if not CLAMP3_ROOT.exists():
        raise FileNotFoundError(f"找不到 CLaMP3 根目录: {CLAMP3_ROOT}")

    if not EXTRACT_MERT.exists():
        raise FileNotFoundError(f"找不到 extract_mert.py: {EXTRACT_MERT}")

    if not CLAMP3_CODE_DIR.exists():
        raise FileNotFoundError(f"找不到 CLaMP3 code 目录: {CLAMP3_CODE_DIR}")

    if not EXTRACT_CLAMP3.exists():
        raise FileNotFoundError(f"找不到 extract_clamp3.py: {EXTRACT_CLAMP3}")


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


# =========================
# MERT extraction
# =========================

def prepare_batch_audio(batch):
    """
    只有当 MERT 缺失时才会使用。
    CLaMP3 的 extract_mert.py 是按文件夹处理的，
    所以每次把一批音频放到临时目录。
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


def build_all_mert_if_needed():
    """
    如果 MERT 已经全部存在，直接跳过。
    如果缺少一部分，才补跑缺失的。
    """
    MERT_DIR.mkdir(parents=True, exist_ok=True)

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
        print("MERT 已全部完成，跳过 MERT 提取。")
        return

    if not RUN_MERT_IF_MISSING:
        raise RuntimeError(
            f"MERT 不完整，缺失 {len(pending)} 个；"
            f"但 RUN_MERT_IF_MISSING=False，所以停止。"
        )

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

def build_clamp3_embd_from_mert():
    """
    关键修改：
    已经有 MERT .npy 后，不再调用 clamp3_embd.py。
    直接进入 tools/basline/clamp3/code 下调用 extract_clamp3.py。
    """
    if not MERT_DIR.exists():
        raise FileNotFoundError(f"找不到 MERT 目录: {MERT_DIR}")

    mert_count = count_npy(MERT_DIR)
    embd_count = count_npy(EMBD_DIR)

    print("=" * 80)
    print(f"MERT_DIR: {MERT_DIR}")
    print(f"EMBD_DIR: {EMBD_DIR}")
    print(f"MERT 文件数量: {mert_count}")
    print(f"已有 CLaMP3 embedding 文件数量: {embd_count}")

    if mert_count == 0:
        raise RuntimeError(f"MERT_DIR 中没有 .npy 文件: {MERT_DIR}")

    if SKIP_EMBD_IF_COMPLETE and embd_count >= mert_count:
        print("CLaMP3 audio embedding 数量已经不少于 MERT 数量，跳过生成。")
        return

    EMBD_DIR.mkdir(parents=True, exist_ok=True)

    # 官方推荐是 accelerate launch extract_clamp3.py ... --get_global
    # 用 python -m accelerate.commands.launch 比直接写 accelerate 更稳，
    # 可以保证使用当前 Anaconda env 里的 accelerate。
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "extract_clamp3.py",
        str(MERT_DIR),
        str(EMBD_DIR),
        "--get_global",
    ]

    print("=" * 80)
    print("开始从 MERT 生成 CLaMP3 audio embedding:")
    print(" ".join(cmd))
    print(f"工作目录: {CLAMP3_CODE_DIR}")

    subprocess.run(
        cmd,
        cwd=str(CLAMP3_CODE_DIR),
        check=True
    )

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
    print(f"CLaMP3 code 目录: {CLAMP3_CODE_DIR}")
    print(f"MERT 目录: {MERT_DIR}")
    print(f"audio embedding 目录: {EMBD_DIR}")
    print(f"临时音频目录: {TMP_AUDIO_DIR}")

    check_required_files()

    build_all_mert_if_needed()
    build_clamp3_embd_from_mert()

    print("CLaMP3 全量构建完成")


if __name__ == "__main__":
    main()