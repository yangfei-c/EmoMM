# -*- coding: utf-8 -*-

import json
import os
import re
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from flask import Flask, request, jsonify


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
TOKEN_JSON = BASE_DIR / "data" / "token_assignments.json"

SPACES_KEY = os.environ["DO_SPACES_KEY"]
SPACES_SECRET = os.environ["DO_SPACES_SECRET"]
SPACES_REGION = os.environ.get("DO_SPACES_REGION", "sfo3")
SPACES_BUCKET = os.environ.get("DO_SPACES_BUCKET", "emmrmusic")
SPACES_ENDPOINT = os.environ.get(
    "DO_SPACES_ENDPOINT",
    f"https://{SPACES_REGION}.digitaloceanspaces.com"
)

# 用户打开后临时保留 token 的时间。
# 超过这个时间还没有提交，就认为 abandoned，可以重新分配。
ASSIGN_TTL_MINUTES = int(os.environ.get("ASSIGN_TTL_MINUTES", "5"))

ASSIGNED_PREFIX = "user_study/assigned/"
RESPONSES_PREFIX = "user_study/responses/"

s3 = boto3.client(
    "s3",
    region_name=SPACES_REGION,
    endpoint_url=SPACES_ENDPOINT,
    aws_access_key_id=SPACES_KEY,
    aws_secret_access_key=SPACES_SECRET,
)

allocation_lock = threading.Lock()


@app.get("/")
def health():
    return jsonify({"status": "ok"})


@app.post("/start")
@app.post("/api/start")
def start():
    """
    用户打开总入口时调用。

    正确逻辑：
    1. responses 里已有该 token => 说明这个 token 已经正式完成，不能再分配
    2. assigned 里已有该 token 且未过期 => 说明有人正在做，暂时不能分配
    3. assigned 里已有该 token 但已过期且没有 response => 删除 assigned，释放该 token
    4. 找到可用 token 后写入 assigned marker，返回给前端
    """
    with allocation_lock:
        assignments = load_assignments()
        completed_tokens = list_completed_tokens()

        for user in assignments:
            token = user["token"]

            # 已经提交过 response 的 token，永久不再分配
            if token in completed_tokens:
                continue

            marker = get_assigned_marker(token)

            if marker is not None:
                if is_assigned_marker_expired(marker):
                    # 打开过但长期没提交，释放这个 token
                    delete_assigned_marker(token)
                else:
                    # 正在被别人使用中，暂时跳过
                    continue

            marker_data = {
                "token": token,
                "participant_index": user.get("participant_index", ""),
                "assigned_at": now_iso(),
                "expires_after_minutes": ASSIGN_TTL_MINUTES,
                "user_agent": request.headers.get("User-Agent", ""),
                "ip": get_client_ip()
            }

            put_json_private(assigned_key(token), marker_data)

            return jsonify({
                "ok": True,
                "token": token,
                "participant_index": user.get("participant_index", "")
            })

    return jsonify({
        "error": "所有实验名额已经完成或正在使用中，请稍后再试。"
    }), 409


@app.post("/submit")
@app.post("/api/submit")
def submit():
    """
    用户完成评分后调用。
    每个 token 只允许提交一次。
    """
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    error = validate_response(data)
    if error:
        return jsonify({"error": error}), 400

    token = data["token"]

    # 防止同一个 token 重复提交
    completed_tokens = list_completed_tokens()
    if token in completed_tokens:
        return jsonify({
            "error": "该问卷已经提交过，请不要重复提交。"
        }), 409

    now = now_compact()

    data["received_at"] = now_iso()
    data["user_agent"] = request.headers.get("User-Agent", "")
    data["ip"] = get_client_ip()

    object_key = f"{RESPONSES_PREFIX}{token}_{now}.json"
    put_json_private(object_key, data)

    return jsonify({
        "ok": True,
        "saved_to": object_key
    })


def load_assignments():
    if not TOKEN_JSON.exists():
        raise FileNotFoundError(f"Cannot find {TOKEN_JSON}")

    data = json.loads(TOKEN_JSON.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("token_assignments.json must be a list")

    return data


def list_completed_tokens():
    """
    从 user_study/responses/ 里扫描已提交 token。

    response 文件名格式：
    user_study/responses/P011_UH53KJ7N_20260510T143000Z.json

    注意 token 本身含有下划线，所以不能简单 split("_")。
    """
    completed = set()

    paginator = s3.get_paginator("list_objects_v2")
    pattern = re.compile(r"^(.+)_\d{8}T\d{6}Z\.json$")

    for page in paginator.paginate(Bucket=SPACES_BUCKET, Prefix=RESPONSES_PREFIX):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")

            if not key.endswith(".json"):
                continue

            filename = key.replace(RESPONSES_PREFIX, "", 1)
            m = pattern.match(filename)

            if not m:
                continue

            token = m.group(1)
            completed.add(token)

    return completed


def assigned_key(token):
    return f"{ASSIGNED_PREFIX}{token}.json"


def get_assigned_marker(token):
    """
    如果 assigned marker 存在，读取 JSON。
    不存在则返回 None。
    """
    key = assigned_key(token)

    try:
        obj = s3.get_object(Bucket=SPACES_BUCKET, Key=key)
        body = obj["Body"].read().decode("utf-8")
        return json.loads(body)
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))

        if code in ["404", "NoSuchKey", "NotFound"]:
            return None

        raise


def is_assigned_marker_expired(marker):
    assigned_at = marker.get("assigned_at")

    if not assigned_at:
        return True

    try:
        dt = datetime.fromisoformat(assigned_at)
    except ValueError:
        return True

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    return now - dt > timedelta(minutes=ASSIGN_TTL_MINUTES)


def delete_assigned_marker(token):
    key = assigned_key(token)

    try:
        s3.delete_object(Bucket=SPACES_BUCKET, Key=key)
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))

        if code in ["404", "NoSuchKey", "NotFound"]:
            return

        raise


def put_json_private(key, data):
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

    s3.put_object(
        Bucket=SPACES_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
        ACL="private"
    )


def validate_response(data):
    if not data.get("token"):
        return "Missing token"

    if "participant_index" not in data:
        return "Missing participant_index"

    if "trials" not in data or not isinstance(data["trials"], list):
        return "Missing trials"

    if len(data["trials"]) == 0:
        return "Empty trials"

    for trial in data["trials"]:
        if not trial.get("trial_index"):
            return "Missing trial_index"

        if not trial.get("query_id"):
            return "Missing query_id"

        if not trial.get("strategy"):
            return "Missing strategy"

        if not trial.get("preference"):
            return "Missing preference"

        if trial["preference"] not in ["A", "B", "C"]:
            return "Invalid preference"

        ratings = trial.get("ratings", {})

        for label in ["A", "B", "C"]:
            if label not in ratings:
                return f"Missing ratings for method {label}"

            r = ratings[label]

            for field in ["regulation_fit", "helpfulness", "satisfaction"]:
                value = r.get(field)

                if value in [None, ""]:
                    return f"Missing {field} for method {label}"

                try:
                    score = int(value)
                except ValueError:
                    return f"Invalid score: {field}"

                if score < 1 or score > 7:
                    return f"Score out of range: {field}"

    return ""


def get_client_ip():
    forwarded = request.headers.get("X-Forwarded-For", "")

    if forwarded:
        return forwarded.split(",")[0].strip()

    return request.remote_addr or ""


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def now_compact():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)