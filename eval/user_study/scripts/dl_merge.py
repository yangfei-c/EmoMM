# -*- coding: utf-8 -*-

import csv
import json
import os
from pathlib import Path

import boto3


SPACES_KEY = "DO801HBPLEHUEQZFKVXV"
SPACES_SECRET = "wqIPqYRPHVkHkjVcYV4XHvfyZCbXQvSHj7wKTx9HIZc"
SPACES_REGION = "sfo3"
SPACES_BUCKET = "emmrmusic"
SPACES_ENDPOINT = "https://sfo3.digitaloceanspaces.com"

PREFIX = "user_study/responses/"

ROOT = Path(__file__).resolve().parents[3]
MATERIALS_JSON = ROOT / "eval" / "user_study" / "web" / "data" / "materials.json"

OUT_DIR = ROOT / "eval" / "user_study" / "downloaded_responses"
RAW_DIR = OUT_DIR / "raw"
OUT_CSV = OUT_DIR / "responses_merged.csv"


s3 = boto3.client(
    "s3",
    region_name=SPACES_REGION,
    endpoint_url=SPACES_ENDPOINT,
    aws_access_key_id=SPACES_KEY,
    aws_secret_access_key=SPACES_SECRET,
)


def load_method_map():
    """
    用 materials.json 把 A/B/C 还原成真实方法：
    token + trial_index + A/B/C -> ours / clap / clamp3
    """
    method_map = {}

    if not MATERIALS_JSON.exists():
        print("Warning: materials.json not found. method_name will be empty.")
        return method_map

    materials = json.loads(MATERIALS_JSON.read_text(encoding="utf-8"))

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

    for page in paginator.paginate(Bucket=SPACES_BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.endswith(".json"):
                keys.append(key)

    return keys


def download_files(keys):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    local_files = []

    for key in keys:
        filename = key.replace("/", "__")
        local_path = RAW_DIR / filename

        s3.download_file(SPACES_BUCKET, key, str(local_path))
        local_files.append((key, local_path))

    return local_files


def merge_to_csv(local_files, method_map):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "token",
        "participant_index",
        "submitted_at",
        "received_at",
        "trial_index",
        "query_id",
        "strategy",
        "method_label",
        "method_name",
        "regulation_fit",
        "helpfulness",
        "satisfaction",
        "preference",
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
        ip = data.get("ip", "")
        user_agent = data.get("user_agent", "")

        for trial in data.get("trials", []):
            trial_index = trial.get("trial_index", "")
            query_id = trial.get("query_id", "")
            strategy = trial.get("strategy", "")
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
                    "trial_index": trial_index,
                    "query_id": query_id,
                    "strategy": strategy,
                    "method_label": label,
                    "method_name": method_name,
                    "regulation_fit": r.get("regulation_fit", ""),
                    "helpfulness": r.get("helpfulness", ""),
                    "satisfaction": r.get("satisfaction", ""),
                    "preference": preference,
                    "ip": ip,
                    "user_agent": user_agent,
                })

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

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