# -*- coding: utf-8 -*-

import json
import re
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from flask import Flask, request, jsonify


BASE_DIR = Path(__file__).resolve().parent
TOKEN_JSON = BASE_DIR / "data" / "token_assignments.json"

SPACES_KEY = "DO801HBPLEHUEQZFKVXV"
SPACES_SECRET = "wqIPqYRPHVkHkjVcYV4XHvfyZCbXQvSHj7wKTx9HIZc"
SPACES_REGION = "sfo3"
SPACES_BUCKET = "emmrmusic"
SPACES_ENDPOINT = "https://sfo3.digitaloceanspaces.com"

ASSIGN_TTL_MINUTES = 5
ASSIGNED_PREFIX = "user_study/assigned/"
RESPONSES_PREFIX = "user_study/responses/"

SURVEY_VERSION = "v2_single_song_single_rating"
RATING_FIELD = "regulation_suitability"
VALID_LABELS = ["A", "B", "C"]
LOCAL_PORT = 8080


app = Flask(__name__)
lock = threading.Lock()

s3 = boto3.client(
    "s3",
    region_name=SPACES_REGION,
    endpoint_url=SPACES_ENDPOINT,
    aws_access_key_id=SPACES_KEY,
    aws_secret_access_key=SPACES_SECRET,
    config=Config(signature_version="s3v4"),
)


@app.get("/")
def health():
    return jsonify({"status": "ok", "survey_version": SURVEY_VERSION})


@app.post("/start")
@app.post("/api/start")
def start():
    with lock:
        assignments = load_assignments()
        completed = list_completed_tokens()

        for user in assignments:
            token = user["token"]

            if token in completed:
                continue

            marker = get_assigned_marker(token)

            if marker and not marker_expired(marker):
                continue

            if marker:
                delete_assigned_marker(token)

            put_json(assigned_key(token), {
                "token": token,
                "participant_index": user.get("participant_index", ""),
                "assigned_at": now_iso(),
                "expires_after_minutes": ASSIGN_TTL_MINUTES,
                "survey_version": SURVEY_VERSION,
                "user_agent": request.headers.get("User-Agent", ""),
                "ip": get_client_ip(),
            })

            return jsonify({
                "ok": True,
                "token": token,
                "participant_index": user.get("participant_index", ""),
                "survey_version": SURVEY_VERSION,
            })

    return jsonify({"error": "所有实验名额已经完成或正在使用中，请稍后再试。"}), 409


@app.post("/submit")
@app.post("/api/submit")
def submit():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    error = validate_response(data)
    if error:
        return jsonify({"error": error}), 400

    token = data["token"]

    if token in list_completed_tokens():
        return jsonify({"error": "该问卷已经提交过，请不要重复提交。"}), 409

    ts = now_compact()
    key = f"{RESPONSES_PREFIX}{token}_{ts}.json"

    data["received_at"] = now_iso()
    data["server_survey_version"] = SURVEY_VERSION
    data["user_agent"] = request.headers.get("User-Agent", "")
    data["ip"] = get_client_ip()

    put_json(key, data)

    try:
        delete_assigned_marker(token)
    except Exception:
        pass

    return jsonify({"ok": True, "saved_to": key, "survey_version": SURVEY_VERSION})


def load_assignments():
    if not TOKEN_JSON.exists():
        raise FileNotFoundError(TOKEN_JSON)

    data = json.loads(TOKEN_JSON.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("token_assignments.json must be a list")

    return data


def list_completed_tokens():
    tokens = set()
    pattern = re.compile(r"^(.+)_\d{8}T\d{6}Z\.json$")

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=SPACES_BUCKET, Prefix=RESPONSES_PREFIX):
        for obj in page.get("Contents", []):
            filename = obj["Key"].replace(RESPONSES_PREFIX, "", 1)
            match = pattern.match(filename)

            if match:
                tokens.add(match.group(1))

    return tokens


def assigned_key(token):
    return f"{ASSIGNED_PREFIX}{token}.json"


def get_assigned_marker(token):
    try:
        obj = s3.get_object(Bucket=SPACES_BUCKET, Key=assigned_key(token))
        return json.loads(obj["Body"].read().decode("utf-8"))
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ["404", "NoSuchKey", "NotFound"]:
            return None
        raise


def marker_expired(marker):
    try:
        assigned_at = datetime.fromisoformat(marker["assigned_at"])
    except Exception:
        return True

    if assigned_at.tzinfo is None:
        assigned_at = assigned_at.replace(tzinfo=timezone.utc)

    return datetime.now(timezone.utc) - assigned_at > timedelta(minutes=ASSIGN_TTL_MINUTES)


def delete_assigned_marker(token):
    try:
        s3.delete_object(Bucket=SPACES_BUCKET, Key=assigned_key(token))
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code not in ["404", "NoSuchKey", "NotFound"]:
            raise


def put_json(key, data):
    s3.put_object(
        Bucket=SPACES_BUCKET,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json; charset=utf-8",
        ACL="private",
    )


def validate_response(data):
    if not data.get("token"):
        return "Missing token"

    if "participant_index" not in data:
        return "Missing participant_index"

    trials = data.get("trials")
    if not isinstance(trials, list) or not trials:
        return "Missing trials"

    for trial in trials:
        if not trial.get("trial_index"):
            return "Missing trial_index"

        if not trial.get("query_id"):
            return "Missing query_id"

        if not trial.get("strategy"):
            return "Missing strategy"

        if trial.get("preference") not in VALID_LABELS:
            return "Invalid preference"

        ratings = trial.get("ratings")
        if not isinstance(ratings, dict):
            return "Invalid ratings"

        for label in VALID_LABELS:
            if label not in ratings:
                return f"Missing ratings for method {label}"

            value = ratings[label].get(RATING_FIELD)
            error = validate_score(value, f"{label}.{RATING_FIELD}")

            if error:
                return error

    return ""


def validate_score(value, field):
    if value in [None, ""]:
        return f"Missing {field}"

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
    app.run(host="0.0.0.0", port=LOCAL_PORT)