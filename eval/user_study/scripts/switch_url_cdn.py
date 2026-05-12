# -*- coding: utf-8 -*-

import json
from pathlib import Path


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

MATERIALS_JSON = ROOT / "eval" / "user_study" / "web" / "data" / "materials.json"
AUDIO_URL_JSON = ROOT / "eval" / "user_study" / "data" / "audio_clip_urls.json"

OLD = "https://emmrmusic.sfo3.digitaloceanspaces.com"
NEW = "https://emmrmusic.sfo3.cdn.digitaloceanspaces.com"


def replace_obj(x):
    if isinstance(x, dict):
        return {k: replace_obj(v) for k, v in x.items()}

    if isinstance(x, list):
        return [replace_obj(v) for v in x]

    if isinstance(x, str):
        return x.replace(OLD, NEW)

    return x


def main():
    for path in [MATERIALS_JSON, AUDIO_URL_JSON]:
        if not path.exists():
            raise FileNotFoundError(f"找不到文件: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        data = replace_obj(data)

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"已替换为 CDN URL: {path}")

    print("全部完成。")


if __name__ == "__main__":
    main()