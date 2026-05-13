# -*- coding: utf-8 -*-

import csv
import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

ASSIGN_JSON = ROOT / "eval" / "user_study" / "backend" / "data" / "token_assignments.json"

OURS_CSV = ROOT / "eval" / "ours" / "ours_top5.csv"
CLAP_CSV = ROOT / "eval" / "clap" / "clap_top5.csv"
CLAMP3_CSV = ROOT / "eval" / "clamp3" / "clamp3_top5.csv"

AUDIO_URL_JSON = ROOT / "eval" / "user_study" / "data" / "audio_clip_urls.json"
OUT_JSON = ROOT / "eval" / "user_study" / "web" / "data" / "materials.json"

TOP_K_SHOW = 1
CANDIDATE_K = 5

RATING_FIELD = "regulation_suitability"
RATING_PROMPT = "这首音乐适合帮助我从当前情绪调节到目标状态。"
PREFERENCE_PROMPT = "本轮三首音乐中，我最愿意选择哪一首来帮助自己完成情绪调节？"


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def clean(x):
    return str(x or "").strip()


def method_key(x):
    x = clean(x).lower()
    if x in ["ours", "our", "proposed"]:
        return "ours"
    if x == "clap":
        return "clap"
    if x in ["clamp3", "clamp"]:
        return "clamp3"
    raise ValueError(f"未知方法名: {x}")


def load_top5(path, name):
    data = {}
    meta = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            qid = clean(row["query_id"])
            rank = int(row["rank"])

            if rank > CANDIDATE_K:
                continue

            song_id = clean(row["song_id"])
            if not song_id:
                continue

            if qid not in meta:
                meta[qid] = {
                    "query_id": qid,
                    "query_text": clean(row.get("text")),
                    "current_state": clean(row.get("current_state")),
                    "regulation_goal": clean(row.get("regulation_goal")),
                    "strategy": clean(row.get("strategy")),
                    "strategy_zh": clean(row.get("strategy_zh")) or clean(row.get("strategy")),
                    "strength": row.get("strength", ""),
                }

            item = {
                "rank": rank,
                "song_id": song_id,
                "audio_url": "",
                "score": row.get("score", ""),
                "display_tags": row.get("display_tags", ""),
                "va": row.get("va", ""),
            }

            data.setdefault(qid, []).append(item)

    for qid in data:
        data[qid] = sorted(data[qid], key=lambda x: x["rank"])

    print(f"{name}: query={len(data)}, items={sum(len(v) for v in data.values())}")
    return data, meta


def merge_meta(*metas):
    out = {}
    for meta in metas:
        for qid, value in meta.items():
            out.setdefault(qid, value)
    return out


def attach_urls(all_data, url_map):
    missing = []

    for method, qmap in all_data.items():
        for qid, items in qmap.items():
            for item in items:
                sid = item["song_id"]
                if sid not in url_map or not url_map[sid].get("audio_url"):
                    missing.append((method, qid, sid, item["rank"]))
                else:
                    item["audio_url"] = url_map[sid]["audio_url"]

    if missing:
        for x in missing[:20]:
            print("缺少 audio_url:", x)
        raise RuntimeError(f"缺少 audio_url 数量: {len(missing)}")


def choose_one(candidates, used):
    for item in candidates:
        if item["song_id"] not in used:
            used.add(item["song_id"])
            return deepcopy(item)

    item = deepcopy(candidates[0])
    used.add(item["song_id"])
    return item


def build_materials(assignments, all_data, qmeta):
    materials = []

    for user in assignments:
        out_user = {
            "token": user["token"],
            "participant_index": user["participant_index"],
            "trials": [],
        }

        for trial in user["trials"]:
            qid = clean(trial["query_id"])
            meta = qmeta[qid]
            used_in_trial = set()

            trial_out = {
                "trial_index": trial["trial_index"],
                "query_id": qid,
                "query_text": meta["query_text"],
                "current_state": meta["current_state"],
                "regulation_goal": meta["regulation_goal"],
                "strategy": meta["strategy"],
                "strategy_zh": meta["strategy_zh"],
                "strength": meta["strength"],
                "rating_fields": [RATING_FIELD],
                "rating_prompts": {RATING_FIELD: RATING_PROMPT},
                "preference_prompt": PREFERENCE_PROMPT,
                "top_k": TOP_K_SHOW,
            }

            for label in ["A", "B", "C"]:
                m = method_key(trial[f"method_{label}"])
                candidates = all_data[m][qid]
                trial_out[label] = {
                    "method": m,
                    "items": [choose_one(candidates, used_in_trial)],
                }

            out_user["trials"].append(trial_out)

        materials.append(out_user)

    return materials


def check_materials(materials):
    empty = 0
    dup = 0

    for user in materials:
        for trial in user["trials"]:
            seen = set()

            for label in ["A", "B", "C"]:
                items = trial[label]["items"]

                if len(items) != 1:
                    raise RuntimeError(f"{user['token']} {trial['query_id']} {label} 不是 1 首")

                item = items[0]

                if not item.get("audio_url"):
                    empty += 1

                if item["song_id"] in seen:
                    dup += 1

                seen.add(item["song_id"])

            if not trial.get("current_state") or not trial.get("regulation_goal") or not trial.get("strategy_zh"):
                raise RuntimeError(f"{trial['query_id']} query 展示字段缺失")

    if empty:
        raise RuntimeError(f"空 audio_url 数量: {empty}")

    if dup:
        raise RuntimeError(f"同一 trial 内重复歌曲数量: {dup}")


def main():
    assignments = read_json(ASSIGN_JSON)
    url_map = read_json(AUDIO_URL_JSON)

    ours, meta1 = load_top5(OURS_CSV, "ours")
    clap, meta2 = load_top5(CLAP_CSV, "clap")
    clamp3, meta3 = load_top5(CLAMP3_CSV, "clamp3")

    all_data = {
        "ours": ours,
        "clap": clap,
        "clamp3": clamp3,
    }

    attach_urls(all_data, url_map)

    materials = build_materials(
        assignments=assignments,
        all_data=all_data,
        qmeta=merge_meta(meta1, meta2, meta3),
    )

    check_materials(materials)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(materials, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print(f"materials.json 已生成: {OUT_JSON}")
    print(f"用户数: {len(materials)}")
    print(f"trial 数: {sum(len(x['trials']) for x in materials)}")
    print("检查通过。")


if __name__ == "__main__":
    main()