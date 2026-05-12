# -*- coding: utf-8 -*-

import csv
import json
import os
import subprocess
from pathlib import Path

import boto3


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

OURS_CSV = ROOT / "eval" / "ours" / "ours_top5.csv"
CLAP_CSV = ROOT / "eval" / "clap" / "clap_top5.csv"
CLAMP3_CSV = ROOT / "eval" / "clamp3" / "clamp3_top5.csv"

OUT_DIR = Path(r"D:\AAAA_audio_dataset\eval\user_study\clips")
MANIFEST_CSV = ROOT / "eval" / "user_study" / "data" / "clips_manifest.csv"
URL_JSON = ROOT / "eval" / "user_study" / "data" / "audio_clip_urls.json"

TOP_K = 3

# 统一裁剪设置
START_SECONDS = 30
DURATION_SECONDS = 15

# Spaces 上传路径
SPACES_PREFIX = "user_study/audio"


def read_topk_rows(csv_path, method):
    rows = []

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rank = int(row["rank"])

            if rank > TOP_K:
                continue

            song_id = str(row["song_id"]).strip()
            audio_path = str(row.get("audio_path", "")).strip()

            if not song_id:
                continue

            if not audio_path:
                print(f"[跳过] audio_path 为空: {method} {row.get('query_id')} rank={rank} song_id={song_id}")
                continue

            rows.append({
                "method": method,
                "query_id": row["query_id"],
                "rank": rank,
                "song_id": song_id,
                "audio_path": audio_path,
            })

    return rows


def collect_unique_clips():
    all_rows = []
    all_rows.extend(read_topk_rows(OURS_CSV, "ours"))
    all_rows.extend(read_topk_rows(CLAP_CSV, "clap"))
    all_rows.extend(read_topk_rows(CLAMP3_CSV, "clamp3"))

    # song_id 去重；同一首歌只裁剪/上传一次
    clip_map = {}

    for row in all_rows:
        song_id = row["song_id"]

        if song_id not in clip_map:
            clip_map[song_id] = {
                "song_id": song_id,
                "audio_path": row["audio_path"],
                "used_by": [],
            }

        clip_map[song_id]["used_by"].append({
            "method": row["method"],
            "query_id": row["query_id"],
            "rank": row["rank"],
        })

    print("=" * 80)
    print(f"Top-3 总引用数量: {len(all_rows)}")
    print(f"唯一音频数量: {len(clip_map)}")

    return clip_map


def run_ffmpeg(input_path, output_path, start_seconds):
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_seconds),
        "-t", str(DURATION_SECONDS),
        "-i", str(input_path),
        "-vn",
        "-ac", "2",
        "-ar", "44100",
        "-b:a", "160k",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(output_path),
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )


def clip_one(song_id, audio_path):
    src = Path(audio_path)
    dst = OUT_DIR / f"{song_id}.mp3"

    if dst.exists() and dst.stat().st_size > 0:
        return dst

    if not src.exists():
        raise FileNotFoundError(f"原始音频不存在: {src}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        run_ffmpeg(src, dst, START_SECONDS)
    except subprocess.CalledProcessError:
        # 如果音频太短，从 0 秒重新裁剪
        print(f"[重试] {song_id}: 30 秒处裁剪失败，改从 0 秒开始")
        run_ffmpeg(src, dst, 0)

    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"裁剪失败: {song_id}")

    return dst


def get_spaces_client():
    key = "DO801HBPLEHUEQZFKVXV"
    secret = "wqIPqYRPHVkHkjVcYV4XHvfyZCbXQvSHj7wKTx9HIZc"
    region = os.environ.get("DO_SPACES_REGION", "sfo3")
    endpoint = os.environ.get("DO_SPACES_ENDPOINT", f"https://{region}.digitaloceanspaces.com")

    if not key:
        raise RuntimeError("缺少环境变量 DO_SPACES_KEY")
    if not secret:
        raise RuntimeError("缺少环境变量 DO_SPACES_SECRET")

    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )


def upload_one(s3, local_path, song_id):
    bucket = os.environ.get("DO_SPACES_BUCKET", "emmrmusic")
    region = os.environ.get("DO_SPACES_REGION", "sfo3")

    object_key = f"{SPACES_PREFIX}/{song_id}.mp3"

    s3.upload_file(
        str(local_path),
        bucket,
        object_key,
        ExtraArgs={
            "ACL": "public-read",
            "ContentType": "audio/mpeg",
            "CacheControl": "public, max-age=31536000",
        },
    )

    url = f"https://{bucket}.{region}.digitaloceanspaces.com/{object_key}"
    return object_key, url


def save_manifest(clip_map, url_map):
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "song_id",
                "audio_path",
                "clip_path",
                "spaces_key",
                "audio_url",
                "used_count",
            ],
        )
        writer.writeheader()

        for song_id, info in sorted(clip_map.items()):
            u = url_map[song_id]

            writer.writerow({
                "song_id": song_id,
                "audio_path": info["audio_path"],
                "clip_path": u["clip_path"],
                "spaces_key": u["spaces_key"],
                "audio_url": u["audio_url"],
                "used_count": len(info["used_by"]),
            })

    URL_JSON.parent.mkdir(parents=True, exist_ok=True)
    URL_JSON.write_text(
        json.dumps(url_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print(f"已保存 manifest: {MANIFEST_CSV}")
    print(f"已保存 audio url map: {URL_JSON}")


def main():
    clip_map = collect_unique_clips()

    s3 = get_spaces_client()

    url_map = {}
    failed = []

    for idx, (song_id, info) in enumerate(sorted(clip_map.items()), start=1):
        print(f"[{idx}/{len(clip_map)}] 处理 {song_id}")

        try:
            clip_path = clip_one(song_id, info["audio_path"])
            spaces_key, audio_url = upload_one(s3, clip_path, song_id)

            url_map[song_id] = {
                "song_id": song_id,
                "audio_path": info["audio_path"],
                "clip_path": str(clip_path),
                "spaces_key": spaces_key,
                "audio_url": audio_url,
            }

        except Exception as e:
            print(f"[失败] {song_id}")
            print(repr(e))
            failed.append({
                "song_id": song_id,
                "audio_path": info["audio_path"],
                "error": repr(e),
            })

    save_manifest(clip_map, url_map)

    print("=" * 80)
    print(f"成功数量: {len(url_map)}")
    print(f"失败数量: {len(failed)}")

    if failed:
        fail_path = ROOT / "eval" / "user_study" / "data" / "clips_failed.json"
        fail_path.write_text(
            json.dumps(failed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"失败记录: {fail_path}")
        raise RuntimeError("存在裁剪或上传失败，请先处理失败项。")

    print("全部 clips 裁剪并上传完成。")


if __name__ == "__main__":
    main()