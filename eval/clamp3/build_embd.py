# -*- coding: utf-8 -*-

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

CLAMP3_ROOT = ROOT / "tools" / "basline" / "clamp3"
CODE_DIR = CLAMP3_ROOT / "code"
EXTRACT_CLAMP3 = CODE_DIR / "extract_clamp3.py"

BASE_DIR = Path(r"D:\AAAA_audio_dataset\eval\clamp3")

QUERY_TXT = BASE_DIR / "query_txt"
MERT_DIR = BASE_DIR / "mert"

QUERY_EMBD = BASE_DIR / "query_embd"
AUDIO_EMBD = BASE_DIR / "audio_embd"


def run_extract(input_dir, output_dir):
    if output_dir.exists():
        shutil.rmtree(output_dir)

    cmd = [
        sys.executable,
        str(EXTRACT_CLAMP3),
        str(input_dir),
        str(output_dir),
        "--get_global",
    ]

    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(CODE_DIR), check=True)


def main():
    print("生成 query embedding...")
    run_extract(QUERY_TXT, QUERY_EMBD)

    print("生成 audio embedding...")
    run_extract(MERT_DIR, AUDIO_EMBD)

    print("完成")
    print(f"query_embd: {QUERY_EMBD}")
    print(f"audio_embd: {AUDIO_EMBD}")


if __name__ == "__main__":
    main()