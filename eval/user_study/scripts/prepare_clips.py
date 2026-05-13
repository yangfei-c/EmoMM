# -*- coding: utf-8 -*-

import csv
import json
import shutil
import subprocess
from pathlib import Path

import boto3
from botocore.client import Config


# =============================================================================
# 固定配置区：不从环境变量读取
# =============================================================================

ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

OURS_CSV = ROOT / "eval" / "ours" / "ours_top5.csv"
CLAP_CSV = ROOT / "eval" / "clap" / "clap_top5.csv"
CLAMP3_CSV = ROOT / "eval" / "clamp3" / "clamp3_top5.csv"

# 本地裁剪音频保存目录
OUT_DIR = Path(r"D:\AAAA_audio_dataset\eval\user_study\clips")

# 记录裁剪/上传清单
MANIFEST_CSV = ROOT / "eval" / "user_study" / "data" / "clips_manifest.csv"

# build_materials.py 读取这个文件来给 song_id 挂 audio_url
URL_JSON = ROOT / "eval" / "user_study" / "data" / "audio_clip_urls.json"

# 新版 materials 仍然会读取 Top-5 候选，以便 A/B/C 撞歌时用 rank2-rank5 兜底
TOP_K_FOR_UPLOAD = 5

# 裁剪参数
START_SECONDS = 30
DURATION_SECONDS = 15

# ffmpeg 参数
AUDIO_BITRATE = "160k"
AUDIO_SAMPLE_RATE = "44100"
AUDIO_CHANNELS = "2"

# DigitalOcean Spaces 配置
# 注意：这里请你在本地填入自己的新 key / secret，不要提交 GitHub。
SPACES_KEY = "DO801HBPLEHUEQZFKVXV"
SPACES_SECRET = "wqIPqYRPHVkHkjVcYV4XHvfyZCbXQvSHj7wKTx9HIZc"

SPACES_BUCKET = "emmrmusic"
SPACES_REGION = "sfo3"

# S3 API endpoint：用于 boto3 上传
SPACES_ENDPOINT = "https://sfo3.digitaloceanspaces.com"

# 公开访问前缀：用于写入 audio_clip_urls.json
SPACES_PUBLIC_BASE = "https://emmrmusic.sfo3.digitaloceanspaces.com"

# Spaces 对象目录
SPACES_PREFIX = "user_study/audio"

# 是否跳过 audio_url 已存在的 song_id
SKIP_IF_URL_EXISTS = True

# 是否跳过本地 clip 已经存在的 song_id
SKIP_CUT_IF_CLIP_EXISTS = True


# =============================================================================
# CSV 读取
# =============================================================================

def read_topk_rows(csv_path, method):
    rows = []

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {"query_id", "rank", "song_id", "audio_path"}
        missing_cols = required_cols - set(reader.fieldnames or [])

        if missing_cols:
            raise RuntimeError(f"{csv_path} 缺少必要列: {missing_cols}")

        for row in reader:
            rank = int(row["rank"])

            if rank > TOP_K_FOR_UPLOAD:
                continue

            song_id = str(row["song_id"]).strip()
            audio_path = str(row.get("audio_path", "")).strip()

            if not song_id:
                continue

            if not audio_path:
                print(
                    f"[跳过] audio_path 为空: "
                    f"{method} {row.get('query_id')} rank={rank} song_id={song_id}"
                )
                continue

            rows.append({
                "method": method,
                "query_id": str(row["query_id"]).strip(),
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

        # 如果第一次遇到的 audio_path 为空，后面遇到非空则补上
        if not clip_map[song_id].get("audio_path") and row["audio_path"]:
            clip_map[song_id]["audio_path"] = row["audio_path"]

    print("=" * 80)
    print(f"Top-{TOP_K_FOR_UPLOAD} 总引用数量: {len(all_rows)}")
    print(f"唯一音频数量: {len(clip_map)}")

    return clip_map


# =============================================================================
# ffmpeg 裁剪
# =============================================================================

def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        raise RuntimeError("没有找到 ffmpeg。请先确认命令行可以运行: ffmpeg -version")


def run_ffmpeg(input_path, output_path, start_seconds):
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-t",
        str(DURATION_SECONDS),
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        AUDIO_CHANNELS,
        "-ar",
        AUDIO_SAMPLE_RATE,
        "-b:a",
        AUDIO_BITRATE,
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
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

    if SKIP_CUT_IF_CLIP_EXISTS and dst.exists() and dst.stat().st_size > 0:
        return dst

    if not src.exists():
        raise FileNotFoundError(f"原始音频不存在: {src}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        run_ffmpeg(src, dst, START_SECONDS)
    except subprocess.CalledProcessError:
        print(f"[重试] {song_id}: {START_SECONDS} 秒处裁剪失败，改从 0 秒开始")
        run_ffmpeg(src, dst, 0)

    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"裁剪失败: {song_id}")

    return dst


# =============================================================================
# Spaces 上传
# =============================================================================

def validate_spaces_config():
    if not SPACES_KEY or "在这里填" in SPACES_KEY:
        raise RuntimeError("请先在脚本顶部填写 SPACES_KEY")

    if not SPACES_SECRET or "在这里填" in SPACES_SECRET:
        raise RuntimeError("请先在脚本顶部填写 SPACES_SECRET")

    if not SPACES_BUCKET:
        raise RuntimeError("SPACES_BUCKET 为空")

    if not SPACES_REGION:
        raise RuntimeError("SPACES_REGION 为空")

    if not SPACES_ENDPOINT:
        raise RuntimeError("SPACES_ENDPOINT 为空")

    if not SPACES_PUBLIC_BASE:
        raise RuntimeError("SPACES_PUBLIC_BASE 为空")


def get_spaces_client():
    validate_spaces_config()

    return boto3.client(
        "s3",
        region_name=SPACES_REGION,
        endpoint_url=SPACES_ENDPOINT,
        aws_access_key_id=SPACES_KEY,
        aws_secret_access_key=SPACES_SECRET,
        config=Config(signature_version="s3v4"),
    )


def upload_one(s3, local_path, song_id):
    object_key = f"{SPACES_PREFIX.strip('/')}/{song_id}.mp3"

    s3.upload_file(
        str(local_path),
        SPACES_BUCKET,
        object_key,
        ExtraArgs={
            "ACL": "public-read",
            "ContentType": "audio/mpeg",
            "CacheControl": "public, max-age=31536000",
        },
    )

    url = f"{SPACES_PUBLIC_BASE.rstrip('/')}/{object_key}"

    return object_key, url


# =============================================================================
# URL map / manifest
# =============================================================================

def normalize_url_map(raw):
    """
    兼容旧格式：
    {
      "track_x": {"audio_url": "..."}
    }

    也兼容简单格式：
    {
      "track_x": "https://..."
    }
    """
    out = {}

    if not isinstance(raw, dict):
        return out

    for song_id, value in raw.items():
        sid = str(song_id).strip()

        if not sid:
            continue

        if isinstance(value, dict):
            out[sid] = value
        else:
            out[sid] = {
                "song_id": sid,
                "audio_url": str(value),
            }

    return out


def load_existing_url_map():
    if URL_JSON.exists():
        try:
            raw = json.loads(URL_JSON.read_text(encoding="utf-8"))
            return normalize_url_map(raw)
        except Exception as e:
            print(f"[警告] 读取旧 audio_clip_urls.json 失败，将使用空 map: {repr(e)}")
            return {}

    return {}


def backup_file(path):
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"已备份旧文件: {backup}")


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
                "space_key",
                "audio_url",
                "used_count",
                "used_by",
            ],
        )
        writer.writeheader()

        for song_id, info in sorted(clip_map.items()):
            if song_id not in url_map:
                continue

            u = url_map[song_id]

            writer.writerow({
                "song_id": song_id,
                "audio_path": info["audio_path"],
                "clip_path": u.get("clip_path", ""),
                "spaces_key": u.get("spaces_key", u.get("space_key", "")),
                "space_key": u.get("space_key", u.get("spaces_key", "")),
                "audio_url": u.get("audio_url", ""),
                "used_count": len(info["used_by"]),
                "used_by": json.dumps(info["used_by"], ensure_ascii=False),
            })

    URL_JSON.parent.mkdir(parents=True, exist_ok=True)
    backup_file(URL_JSON)

    URL_JSON.write_text(
        json.dumps(url_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print(f"已保存 manifest: {MANIFEST_CSV}")
    print(f"已保存 audio url map: {URL_JSON}")


def save_failed(failed):
    fail_path = ROOT / "eval" / "user_study" / "data" / "clips_failed.json"
    fail_path.parent.mkdir(parents=True, exist_ok=True)

    fail_path.write_text(
        json.dumps(failed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"失败记录: {fail_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("新版用户研究 Top-5 clips 裁剪并上传开始")
    print(f"ROOT: {ROOT}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"URL_JSON: {URL_JSON}")
    print(f"SPACES_BUCKET: {SPACES_BUCKET}")
    print(f"SPACES_REGION: {SPACES_REGION}")
    print(f"SPACES_ENDPOINT: {SPACES_ENDPOINT}")
    print(f"SPACES_PUBLIC_BASE: {SPACES_PUBLIC_BASE}")
    print(f"SPACES_PREFIX: {SPACES_PREFIX}")
    print(f"TOP_K_FOR_UPLOAD: {TOP_K_FOR_UPLOAD}")
    print(f"START_SECONDS: {START_SECONDS}")
    print(f"DURATION_SECONDS: {DURATION_SECONDS}")

    check_ffmpeg()
    validate_spaces_config()

    clip_map = collect_unique_clips()
    s3 = get_spaces_client()

    url_map = load_existing_url_map()
    failed = []

    existing_count = 0
    uploaded_count = 0
    clipped_count = 0

    for idx, (song_id, info) in enumerate(sorted(clip_map.items()), start=1):
        print("-" * 80)
        print(f"[{idx}/{len(clip_map)}] 处理 {song_id}")

        try:
            if (
                SKIP_IF_URL_EXISTS
                and song_id in url_map
                and url_map[song_id].get("audio_url")
            ):
                print(f"  已存在 audio_url，跳过裁剪/上传: {song_id}")
                existing_count += 1
                continue

            clip_path = clip_one(song_id, info["audio_path"])
            clipped_count += 1

            spaces_key, audio_url = upload_one(s3, clip_path, song_id)
            uploaded_count += 1

            url_map[song_id] = {
                "song_id": song_id,
                "audio_path": info["audio_path"],
                "source_audio_path": info["audio_path"],
                "clip_path": str(clip_path),

                # 两个字段都保留，兼容旧脚本/新脚本
                "spaces_key": spaces_key,
                "space_key": spaces_key,

                "audio_url": audio_url,
                "clip_start_seconds": START_SECONDS,
                "clip_duration_seconds": DURATION_SECONDS,
                "used_by": info["used_by"],
            }

            print(f"  上传成功: {audio_url}")

        except Exception as e:
            print(f"[失败] {song_id}")
            print(repr(e))
            failed.append({
                "song_id": song_id,
                "audio_path": info.get("audio_path", ""),
                "error": repr(e),
                "used_by": info.get("used_by", []),
            })

    save_manifest(clip_map, url_map)

    print("=" * 80)
    print("处理完成")
    print(f"audio_url 总数量: {len(url_map)}")
    print(f"本次已存在跳过数量: {existing_count}")
    print(f"本次裁剪数量: {clipped_count}")
    print(f"本次上传数量: {uploaded_count}")
    print(f"失败数量: {len(failed)}")

    if failed:
        save_failed(failed)
        raise RuntimeError("存在裁剪或上传失败，请先处理失败项。")

    print("全部新版 Top-5 clips 裁剪并上传完成。")


if __name__ == "__main__":
    main()