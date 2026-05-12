# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

ASSIGN_JSON = ROOT / "eval" / "user_study" / "data" / "token_assignments.json"

OURS_CSV = ROOT / "eval" / "ours" / "ours_top5.csv"
CLAP_CSV = ROOT / "eval" / "clap" / "clap_top5.csv"
CLAMP3_CSV = ROOT / "eval" / "clamp3" / "clamp3_top5.csv"

AUDIO_URL_JSON = ROOT /"eval"/"user_study" / "data" / "audio_clip_urls.json"

OUT_JSON = ROOT /"eval"/ "user_study" / "web" / "data" / "materials.json"

TOP_K = 3


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_topk_csv(csv_path, method_name):
    """
    读取某个方法的 top5.csv，只保留 Top-3。
    返回:
    {
        "q01": [item1, item2, item3],
        "q02": [...]
    }
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    data = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            qid = str(row["query_id"]).strip()
            rank = int(row["rank"])
            song_id = str(row["song_id"]).strip()

            if rank > TOP_K:
                continue

            item = {
                "rank": rank,
                "song_id": song_id,
                "audio_url": "",

                # 这些字段保留给后续分析，不在前端显示
                "score": row.get("score", ""),
                "display_tags": row.get("display_tags", ""),
                "va": row.get("va", ""),
            }

            data.setdefault(qid, []).append(item)

    for qid in data:
        data[qid] = sorted(data[qid], key=lambda x: x["rank"])

    print(f"{method_name}: query 数量={len(data)}, item 数量={sum(len(v) for v in data.values())}")

    return data


def normalize_method_name(method):
    m = str(method).strip().lower()

    if m in {"ours", "our", "proposed"}:
        return "ours"

    if m == "clap":
        return "clap"

    if m in {"clamp3", "clamp"}:
        return "clamp3"

    raise ValueError(f"未知 method 名称: {method}")


def attach_audio_urls(method_data, audio_url_map):
    """
    给每个 item 写入 Spaces audio_url。
    """
    missing = []

    for method_name, q_map in method_data.items():
        for qid, items in q_map.items():
            for item in items:
                song_id = item["song_id"]

                if song_id not in audio_url_map:
                    missing.append({
                        "method": method_name,
                        "query_id": qid,
                        "song_id": song_id,
                    })
                    continue

                item["audio_url"] = audio_url_map[song_id]["audio_url"]

    if missing:
        print("以下歌曲缺少 audio_url，前 20 个如下：")
        for x in missing[:20]:
            print(x)

        raise RuntimeError(f"缺少 audio_url 数量: {len(missing)}")


def get_items_for_trial(method_name, qid, method_data):
    key = normalize_method_name(method_name)

    if qid not in method_data[key]:
        raise KeyError(f"{key} 缺少 query_id={qid}")

    items = method_data[key][qid]

    if len(items) < TOP_K:
        raise RuntimeError(f"{key} 的 {qid} 只有 {len(items)} 个 item，不足 Top-{TOP_K}")

    return items[:TOP_K]


def build_materials(assignments, method_data):
    materials = []

    for user in assignments:
        user_item = {
            "token": user["token"],
            "participant_index": user["participant_index"],
            "trials": [],
        }

        for trial in user["trials"]:
            qid = str(trial["query_id"]).strip()

            method_a = trial["method_A"]
            method_b = trial["method_B"]
            method_c = trial["method_C"]

            trial_item = {
                "trial_index": trial["trial_index"],
                "query_id": qid,
                "strategy": trial["strategy"],
                "query_text": trial["query_text"],

                # method 保留在 JSON 中，方便 dl_merge.py 还原真实方法；
                # 前端 app.js 不要显示 method 字段。
                "A": {
                    "method": method_a,
                    "items": get_items_for_trial(method_a, qid, method_data),
                },
                "B": {
                    "method": method_b,
                    "items": get_items_for_trial(method_b, qid, method_data),
                },
                "C": {
                    "method": method_c,
                    "items": get_items_for_trial(method_c, qid, method_data),
                },

                "rating_fields": [
                    "regulation_fit",
                    "helpfulness",
                    "satisfaction",
                ],
                "top_k": TOP_K,
            }

            user_item["trials"].append(trial_item)

        materials.append(user_item)

    return materials


def check_materials(materials):
    empty_url = 0
    item_count = 0
    trial_count = 0

    for user in materials:
        for trial in user["trials"]:
            trial_count += 1

            for arm in ["A", "B", "C"]:
                items = trial[arm]["items"]

                if len(items) != TOP_K:
                    raise RuntimeError(
                        f"{user['token']} {trial['query_id']} {arm} item 数量不是 {TOP_K}"
                    )

                for item in items:
                    item_count += 1

                    if not item.get("audio_url"):
                        empty_url += 1

    print("=" * 80)
    print(f"用户数量: {len(materials)}")
    print(f"trial 数量: {trial_count}")
    print(f"音频 item 引用数量: {item_count}")
    print(f"空 audio_url 数量: {empty_url}")

    if empty_url > 0:
        raise RuntimeError("materials.json 中存在空 audio_url")

    print("materials.json 检查通过。")


def main():
    assignments = read_json(ASSIGN_JSON)
    audio_url_map = read_json(AUDIO_URL_JSON)

    method_data = {
        "ours": load_topk_csv(OURS_CSV, "ours"),
        "clap": load_topk_csv(CLAP_CSV, "clap"),
        "clamp3": load_topk_csv(CLAMP3_CSV, "clamp3"),
    }

    attach_audio_urls(method_data, audio_url_map)

    materials = build_materials(assignments, method_data)
    check_materials(materials)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(materials, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print(f"最终 materials.json 已生成:")
    print(OUT_JSON)


if __name__ == "__main__":
    main()