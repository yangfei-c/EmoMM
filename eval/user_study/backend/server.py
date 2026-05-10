# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime, timezone

import boto3
from flask import Flask, request, jsonify


app = Flask(__name__)

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


@app.get("/")
def health():
    return jsonify({"status": "ok"})


@app.post("/submit")
@app.post("/api/submit")
def submit():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    error = validate_response(data)
    if error:
        return jsonify({"error": error}), 400

    object_key = save_to_spaces(data)

    return jsonify({
        "ok": True,
        "saved_to": object_key
    })


def validate_response(data):
    if not data.get("token"):
        return "Missing token"

    if "trials" not in data or not isinstance(data["trials"], list):
        return "Missing trials"

    for trial in data["trials"]:
        if not trial.get("query_id"):
            return "Missing query_id"

        if not trial.get("preference"):
            return "Missing preference"

        ratings = trial.get("ratings", {})

        for label in ["A", "B", "C"]:
            if label not in ratings:
                return f"Missing ratings for method {label}"

            for field in ["regulation_fit", "helpfulness", "satisfaction"]:
                value = ratings[label].get(field)

                if value in [None, ""]:
                    return f"Missing {field} for method {label}"

                try:
                    score = int(value)
                except ValueError:
                    return f"Invalid score: {field}"

                if score < 1 or score > 7:
                    return f"Score out of range: {field}"

    return ""


def save_to_spaces(data):
    token = data["token"]
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    object_key = f"user_study/responses/{token}_{now}.json"

    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

    s3.put_object(
        Bucket=SPACES_BUCKET,
        Key=object_key,
        Body=body,
        ContentType="application/json",
        ACL="private"
    )

    return object_key


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)