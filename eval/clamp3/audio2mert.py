import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

POOL_JSON = ROOT / "eval" / "candidate_pool.json"

CLAMP3_ROOT = ROOT / "tools" / "basline" / "clamp3"
EXTRACT_MERT = CLAMP3_ROOT / "preprocessing" / "audio" / "extract_mert.py"

BASE_DIR = Path(r"D:\AAAA_audio_dataset\eval\clamp3")
AUDIO_TMP = BASE_DIR / "_audio_tmp"
MERT_DIR = BASE_DIR / "mert"


def main():
    pool = json.loads(POOL_JSON.read_text(encoding="utf-8"))

    if AUDIO_TMP.exists():
        shutil.rmtree(AUDIO_TMP)
    AUDIO_TMP.mkdir(parents=True, exist_ok=True)

    MERT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0

    for item in pool:
        song_id = str(item["song_id"])
        src = Path(str(item["song_path"]))

        if not src.exists():
            continue

        ext = src.suffix.lower()
        if ext not in [".mp3", ".wav"]:
            continue

        dst = AUDIO_TMP / f"{song_id}{ext}"
        os.link(src, dst)

        count += 1

    print(f"候选音频数量: {count}")
    print(f"开始生成 MERT: {MERT_DIR}")

    cmd = [
        sys.executable,
        str(EXTRACT_MERT),
        "--input_path",
        str(AUDIO_TMP),
        "--output_path",
        str(MERT_DIR),
        "--model_path",
        "m-a-p/MERT-v1-95M",
        "--mean_features",
    ]

    print(" ".join(cmd))

    subprocess.run(cmd, cwd=str(EXTRACT_MERT.parent), check=True)

    shutil.rmtree(AUDIO_TMP)

    print(f"完成 MERT: {MERT_DIR}")


if __name__ == "__main__":
    main()