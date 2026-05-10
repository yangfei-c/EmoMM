# -*- coding: utf-8 -*-

import json
import os
import threading
from datetime import datetime, timezone
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
    后端自动分配一个还没有被占用的 token。
    分配后立即在 Spaces 写入 assigned marker。
    """
    with allocation_lock:
        assignments = load_assignments()

        for user in assignments:
            token = user["token"]
            marker_key = f"user_study/assigned/{token}.json"

            if object_exists(marker_key):
                continue

            marker = {
                "token": token,
                "participant_index": user.get("participant_index", ""),
                "assigned_at": now_iso(),
                "user_agent": request.headers.get("User-Agent", ""),
                "ip": get_client_ip()
            }

            put_json_private(marker_key, marker)

            return jsonify({
                "ok": True,
                "token": token,
                "participant_index": user.get("participant_index", "")
            })

    return jsonify({
        "error": "所有实验名额已经分配完。"
    }), 409


@app.post("/submit")
@app.post("/api/submit")
def submit():
    """
    用户完成评分后调用。
    每次提交保存为一个独立 JSON 文件。
    """
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    error = validate_response(data)
    if error:
        return jsonify({"error": error}), 400

    token = data["token"]
    now = now_compact()

    data["received_at"] = now_iso()
    data["user_agent"] = request.headers.get("User-Agent", "")
    data["ip"] = get_client_ip()

    object_key = f"user_study/responses/{token}_{now}.json"
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


def object_exists(key):
    try:
        s3.head_object(Bucket=SPACES_BUCKET, Key=key)
        return True
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))

        if code in ["404", "NoSuchKey", "NotFound"]:
            return False

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