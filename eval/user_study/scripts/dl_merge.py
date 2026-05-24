# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path

import boto3
from botocore.client import Config


# =============================================================================
# Spaces 配置：本地填入，不要把真实 key/secret 提交到公开 GitHub
# =============================================================================

SPACES_KEY = "DO801HBPLEHUEQZFKVXV"
SPACES_SECRET = "wqIPqYRPHVkHkjVcYV4XHvfyZCbXQvSHj7wKTx9HIZc"

SPACES_REGION = "sfo3"
SPACES_BUCKET = "emmrmusic"
SPACES_ENDPOINT = "https://sfo3.digitaloceanspaces.com"

RESPONSES_PREFIX = "user_study/responses/"

# =============================================================================
# 路径配置
# =============================================================================

ROOT = Path(__file__).resolve().parents[3]

MATERIALS_JSON = ROOT / "eval" / "user_study" / "web" / "data" / "materials.json"

OUT_DIR = ROOT / "eval" / "user_study" / "downloaded_responses"
RAW_DIR = OUT_DIR / "raw"
OUT_CSV = OUT_DIR / "responses_merged.csv"

RATING_FIELD = "regulation_suitability"


s3 = boto3.client(
    "s3",
    region_name=SPACES_REGION,
    endpoint_url=SPACES_ENDPOINT,
    aws_access_key_id=SPACES_KEY,
    aws_secret_access_key=SPACES_SECRET,
    config=Config(signature_version="s3v4"),
)


def load_method_map():
    """
    用 materials.json 把 A/B/C 还原成真实方法：

    token + trial_index + A/B/C -> ours / clap / clamp3
    """
    if not MATERIALS_JSON.exists():
        raise FileNotFoundError(f"找不到 materials.json: {MATERIALS_JSON}")

    materials = json.loads(MATERIALS_JSON.read_text(encoding="utf-8"))
    method_map = {}

    for user in materials:
        token = user["token"]

        for trial in user["trials"]:
            trial_index = trial["trial_index"]

            for label in ["A", "B", "C"]:
                method_map[(token, trial_index, label)] = trial[label]["method"]

    return method_map


def list_response_keys():
    keys = []

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=SPACES_BUCKET, Prefix=RESPONSES_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.endswith(".json"):
                keys.append(key)

    return sorted(keys)


def download_files(keys):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    local_files = []

    for key in keys:
        filename = key.replace("/", "__")
        local_path = RAW_DIR / filename

        s3.download_file(SPACES_BUCKET, key, str(local_path))
        local_files.append((key, local_path))

    return local_files


def is_preferred(preference, label):
    return 1 if preference == label else 0


def make_quality_flag(total_duration_seconds):
    """
    只做疑似异常标记，不自动删除数据。

    旧数据没有 timing 字段时返回空字符串。
    当前只根据打开页面到提交问卷的总耗时判断。
    """
    flags = []

    try:
        total_duration_seconds = float(total_duration_seconds)
    except Exception:
        total_duration_seconds = None

    if total_duration_seconds is not None and total_duration_seconds < 60:
        flags.append("very_short_total_time")

    return "|".join(flags)


def merge_to_csv(local_files, method_map):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "token",
        "participant_index",
        "submitted_at",
        "received_at",
        "server_survey_version",
        "client_version",

        "page_started_at",
        "submit_clicked_at",
        "total_duration_seconds",
        "quality_flag",

        "trial_index",
        "query_id",
        "strategy",
        "strategy_zh",
        "current_state",
        "regulation_goal",

        "method_label",
        "method_name",
        "regulation_suitability",
        "preference",
        "is_preferred",

        "ip",
        "user_agent",
    ]

    rows = []

    for key, path in local_files:
        data = json.loads(path.read_text(encoding="utf-8"))

        token = data.get("token", "")
        participant_index = data.get("participant_index", "")
        submitted_at = data.get("submitted_at", "")
        received_at = data.get("received_at", "")
        server_survey_version = data.get("server_survey_version", "")
        client_version = data.get("version", "")
        ip = data.get("ip", "")
        user_agent = data.get("user_agent", "")

        timing = data.get("timing", {}) or {}

        page_started_at = timing.get("page_started_at", "")
        submit_clicked_at = timing.get("submit_clicked_at", "")
        total_duration_seconds = timing.get("total_duration_seconds", "")

        quality_flag = make_quality_flag(total_duration_seconds)

        for trial in data.get("trials", []):
            trial_index = trial.get("trial_index", "")
            query_id = trial.get("query_id", "")
            strategy = trial.get("strategy", "")
            strategy_zh = trial.get("strategy_zh", "")
            current_state = trial.get("current_state", "")
            regulation_goal = trial.get("regulation_goal", "")
            preference = trial.get("preference", "")

            ratings = trial.get("ratings", {})

            for label in ["A", "B", "C"]:
                r = ratings.get(label, {})
                method_name = method_map.get((token, trial_index, label), "")

                rows.append({
                    "source_file": key,
                    "token": token,
                    "participant_index": participant_index,
                    "submitted_at": submitted_at,
                    "received_at": received_at,
                    "server_survey_version": server_survey_version,
                    "client_version": client_version,

                    "page_started_at": page_started_at,
                    "submit_clicked_at": submit_clicked_at,
                    "total_duration_seconds": total_duration_seconds,
                    "quality_flag": quality_flag,

                    "trial_index": trial_index,
                    "query_id": query_id,
                    "strategy": strategy,
                    "strategy_zh": strategy_zh,
                    "current_state": current_state,
                    "regulation_goal": regulation_goal,

                    "method_label": label,
                    "method_name": method_name,
                    "regulation_suitability": r.get(RATING_FIELD, ""),
                    "preference": preference,
                    "is_preferred": is_preferred(preference, label),

                    "ip": ip,
                    "user_agent": user_agent,
                })

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print(f"Downloaded JSON files: {len(local_files)}")
    print(f"Merged rows: {len(rows)}")
    print(f"CSV saved to: {OUT_CSV}")


def main():
    method_map = load_method_map()
    keys = list_response_keys()

    if not keys:
        print("No response JSON files found.")
        return

    local_files = download_files(keys)
    merge_to_csv(local_files, method_map)


if __name__ == "__main__":
    main()