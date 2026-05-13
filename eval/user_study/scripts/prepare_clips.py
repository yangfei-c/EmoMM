# -*- coding: utf-8 -*-

import csv
import json
import os
import subprocess
from pathlib import Path

import boto3
from botocore.client import Config


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

TOP5_FILES = [
    ("ours", ROOT / "eval" / "ours" / "ours_top5.csv"),
    ("clap", ROOT / "eval" / "clap" / "clap_top5.csv"),
    ("clamp3", ROOT / "eval" / "clamp3" / "clamp3_top5.csv"),
]

OUT_DIR = Path(r"D:\AAAA_audio_dataset\eval\user_study\clips")
URL_JSON = ROOT / "eval" / "user_study" / "data" / "audio_clip_urls.json"

TOP_K_FOR_UPLOAD = 5
START_SECONDS = 30
DURATION_SECONDS = 15

SPACES_KEY = "DO801HBPLEHUEQZFKVXV"
SPACES_SECRET = "wqIPqYRPHVkHkjVcYV4XHvfyZCbXQvSHj7wKTx9HIZc"
SPACES_REGION = "sfo3"
SPACES_BUCKET = "emmrmusic"
SPACES_ENDPOINT = "https://sfo3.digitaloceanspaces.com"
SPACES_PUBLIC_BASE = "https://emmrmusic.sfo3.digitaloceanspaces.com"
SPACES_PREFIX = "user_study/audio"


def read_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def check_key():
    if "在这里填写" in SPACES_KEY or "在这里填写" in SPACES_SECRET:
        raise RuntimeError("请先填写 SPACES_KEY 和 SPACES_SECRET")


def s3_client():
    check_key()
    return boto3.client(
        "s3",
        region_name=SPACES_REGION,
        endpoint_url=SPACES_ENDPOINT,
        aws_access_key_id=SPACES_KEY,
        aws_secret_access_key=SPACES_SECRET,
        config=Config(signature_version="s3v4"),
    )


def collect_items():
    items = {}

    for method, path in TOP5_FILES:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                rank = int(row["rank"])
                if rank > TOP_K_FOR_UPLOAD:
                    continue

                song_id = str(row["song_id"]).strip()
                audio_path = str(row.get("audio_path", "")).strip()

                if not song_id or not audio_path:
                    continue

                items.setdefault(song_id, {
                    "song_id": song_id,
                    "audio_path": audio_path,
                    "used_by": [],
                })

                items[song_id]["used_by"].append({
                    "method": method,
                    "query_id": row.get("query_id", ""),
                    "rank": rank,
                })

    return items


def cut_clip(song_id, audio_path):
    src = Path(audio_path)
    dst = OUT_DIR / f"{song_id}.mp3"

    if dst.exists() and dst.stat().st_size > 0:
        return dst

    if not src.exists():
        raise FileNotFoundError(src)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(START_SECONDS),
        "-t", str(DURATION_SECONDS),
        "-i", str(src),
        "-vn",
        "-ac", "2",
        "-ar", "44100",
        "-b:a", "160k",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(dst),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError:
        cmd[cmd.index(str(START_SECONDS))] = "0"
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)

    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"裁剪失败: {song_id}")

    return dst


def upload(s3, clip_path, song_id):
    key = f"{SPACES_PREFIX}/{song_id}.mp3"

    s3.upload_file(
        str(clip_path),
        SPACES_BUCKET,
        key,
        ExtraArgs={
            "ACL": "public-read",
            "ContentType": "audio/mpeg",
            "CacheControl": "public, max-age=31536000",
        },
    )

    return key, f"{SPACES_PUBLIC_BASE}/{key}"


def main():
    items = collect_items()
    url_map = read_json(URL_JSON)
    s3 = s3_client()

    failed = []

    print(f"唯一音频数量: {len(items)}")
    print(f"已有 URL 数量: {len(url_map)}")

    for i, (song_id, item) in enumerate(sorted(items.items()), start=1):
        print(f"[{i}/{len(items)}] {song_id}")

        if song_id in url_map and url_map[song_id].get("audio_url"):
            print("  已存在，跳过")
            continue

        try:
            clip_path = cut_clip(song_id, item["audio_path"])
            space_key, audio_url = upload(s3, clip_path, song_id)

            url_map[song_id] = {
                "song_id": song_id,
                "audio_path": item["audio_path"],
                "clip_path": str(clip_path),
                "spaces_key": space_key,
                "audio_url": audio_url,
            }

            print("  上传成功")

        except Exception as e:
            print("  失败:", repr(e))
            failed.append({
                "song_id": song_id,
                "audio_path": item["audio_path"],
                "error": repr(e),
            })

    save_json(URL_JSON, url_map)

    print("=" * 80)
    print(f"audio_clip_urls.json 已更新: {URL_JSON}")
    print(f"失败数量: {len(failed)}")

    if failed:
        fail_path = ROOT / "eval" / "user_study" / "data" / "clips_failed.json"
        save_json(fail_path, failed)
        raise RuntimeError(f"有 {len(failed)} 个音频失败，见 {fail_path}")

    print("完成。")


if __name__ == "__main__":
    main()