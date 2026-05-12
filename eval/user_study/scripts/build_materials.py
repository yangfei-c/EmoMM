# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path
from copy import deepcopy


ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec")

ASSIGN_JSON = ROOT / "eval" / "user_study" / "backend" / "data" / "token_assignments.json"

OURS_CSV = ROOT / "eval" / "ours" / "ours_top5.csv"
CLAP_CSV = ROOT / "eval" / "clap" / "clap_top5.csv"
CLAMP3_CSV = ROOT / "eval" / "clamp3" / "clamp3_top5.csv"

AUDIO_URL_JSON = ROOT / "eval" / "user_study" / "data" / "audio_clip_urls.json"

OUT_JSON = ROOT / "eval" / "user_study" / "web" / "data" / "materials.json"

# 页面最终每组展示 3 首
TOP_K_SHOW = 3

# 从 top5.csv 里最多使用前 5 首作为候选，用 rank4/rank5 替换重复歌曲
CANDIDATE_K = 5

# True：同一个 participant 全部 3 个 trial 内，不重复任何 song_id
# False：只在同一个 trial 内避免重复
DEDUP_WITHIN_PARTICIPANT = True


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_method_name(method):
    m = str(method).strip().lower()

    if m in {"ours", "our", "proposed"}:
        return "ours"

    if m == "clap":
        return "clap"

    if m in {"clamp3", "clamp"}:
        return "clamp3"

    raise ValueError(f"未知 method 名称: {method}")


def load_topk_csv(csv_path, method_name):
    """
    读取某个方法的 top5.csv，保留 rank <= CANDIDATE_K。
    返回:
    {
        "q01": [rank1, rank2, ..., rank5],
        ...
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

            if rank > CANDIDATE_K:
                continue

            item = {
                "rank": rank,
                "song_id": song_id,
                "audio_url": "",

                # 保留用于后处理分析；前端不显示这些字段
                "score": row.get("score", ""),
                "display_tags": row.get("display_tags", ""),
                "va": row.get("va", ""),
            }

            data.setdefault(qid, []).append(item)

    for qid in data:
        data[qid] = sorted(data[qid], key=lambda x: x["rank"])

    print(
        f"{method_name}: query 数量={len(data)}, "
        f"候选 item 数量={sum(len(v) for v in data.values())}"
    )

    return data


def attach_audio_urls(method_data, audio_url_map):
    """
    给所有候选 item 写入 Spaces audio_url。
    注意这里包括 rank4/rank5，所以必须先运行 Top-5 版 prepare_audio_clips.py。
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
                        "rank": item["rank"],
                    })
                    continue

                item["audio_url"] = audio_url_map[song_id]["audio_url"]

    if missing:
        print("=" * 80)
        print("以下歌曲缺少 audio_url，前 30 个如下：")
        for x in missing[:30]:
            print(x)

        raise RuntimeError(
            f"缺少 audio_url 数量: {len(missing)}。"
            f"请先运行 Top-5 版 prepare_audio_clips.py。"
        )


def select_items_without_duplicate(candidates, used_song_ids, top_k=TOP_K_SHOW):
    """
    从候选列表中选 top_k 个。
    优先选没有在当前 participant 中出现过的 song_id。
    如果候选不够，再允许重复兜底，保证每组仍然有 3 首。
    """
    selected = []
    selected_ids = set()

    # 第一轮：优先选未在 participant 中出现过，且当前组内不重复的
    for item in candidates:
        sid = item["song_id"]

        if sid in used_song_ids:
            continue

        if sid in selected_ids:
            continue

        selected.append(deepcopy(item))
        selected_ids.add(sid)

        if len(selected) >= top_k:
            break

    # 第二轮兜底：如果不够 3 首，允许用已经在 participant 里出现过的，但组内仍不重复
    if len(selected) < top_k:
        for item in candidates:
            sid = item["song_id"]

            if sid in selected_ids:
                continue

            selected.append(deepcopy(item))
            selected_ids.add(sid)

            if len(selected) >= top_k:
                break

    if len(selected) < top_k:
        raise RuntimeError(
            f"候选不足，无法选出 Top-{top_k}。"
            f"当前候选数量={len(candidates)}, 已选数量={len(selected)}"
        )

    for item in selected:
        used_song_ids.add(item["song_id"])

    return selected


def get_candidates(method_name, qid, method_data):
    key = normalize_method_name(method_name)

    if qid not in method_data[key]:
        raise KeyError(f"{key} 缺少 query_id={qid}")

    candidates = method_data[key][qid]

    if len(candidates) < TOP_K_SHOW:
        raise RuntimeError(
            f"{key} 的 {qid} 只有 {len(candidates)} 个候选，不足 Top-{TOP_K_SHOW}"
        )

    return candidates


def build_materials(assignments, method_data):
    materials = []
    replacement_stats = []

    for user in assignments:
        participant_used_song_ids = set()

        user_item = {
            "token": user["token"],
            "participant_index": user["participant_index"],
            "trials": [],
        }

        for trial in user["trials"]:
            qid = str(trial["query_id"]).strip()

            if DEDUP_WITHIN_PARTICIPANT:
                used_for_this_trial = participant_used_song_ids
            else:
                used_for_this_trial = set()

            arms = {}

            for arm_label in ["A", "B", "C"]:
                method_field = f"method_{arm_label}"
                method_name = trial[method_field]

                candidates = get_candidates(method_name, qid, method_data)

                original_top3 = [x["song_id"] for x in candidates[:TOP_K_SHOW]]

                selected = select_items_without_duplicate(
                    candidates=candidates,
                    used_song_ids=used_for_this_trial,
                    top_k=TOP_K_SHOW,
                )

                selected_ids = [x["song_id"] for x in selected]

                if selected_ids != original_top3:
                    replacement_stats.append({
                        "token": user["token"],
                        "participant_index": user["participant_index"],
                        "trial_index": trial["trial_index"],
                        "query_id": qid,
                        "arm": arm_label,
                        "method": method_name,
                        "original_top3": original_top3,
                        "selected": selected_ids,
                    })

                arms[arm_label] = {
                    "method": method_name,
                    "items": selected,
                }

            trial_item = {
                "trial_index": trial["trial_index"],
                "query_id": qid,
                "strategy": trial["strategy"],
                "query_text": trial["query_text"],

                # 前端只显示 A/B/C，不显示 method
                "A": arms["A"],
                "B": arms["B"],
                "C": arms["C"],

                "rating_fields": [
                    "regulation_fit",
                    "helpfulness",
                    "satisfaction",
                ],
                "top_k": TOP_K_SHOW,
            }

            user_item["trials"].append(trial_item)

        materials.append(user_item)

    return materials, replacement_stats


def check_materials(materials):
    empty_url = 0
    item_count = 0
    trial_count = 0
    within_participant_duplicate_count = 0

    for user in materials:
        seen = set()

        for trial in user["trials"]:
            trial_count += 1

            for arm in ["A", "B", "C"]:
                items = trial[arm]["items"]

                if len(items) != TOP_K_SHOW:
                    raise RuntimeError(
                        f"{user['token']} {trial['query_id']} {arm} item 数量不是 {TOP_K_SHOW}"
                    )

                for item in items:
                    item_count += 1

                    if not item.get("audio_url"):
                        empty_url += 1

                    sid = item["song_id"]

                    if sid in seen:
                        within_participant_duplicate_count += 1

                    seen.add(sid)

    print("=" * 80)
    print(f"用户数量: {len(materials)}")
    print(f"trial 数量: {trial_count}")
    print(f"音频 item 引用数量: {item_count}")
    print(f"空 audio_url 数量: {empty_url}")
    print(f"同一 participant 内重复 song_id 数量: {within_participant_duplicate_count}")

    if empty_url > 0:
        raise RuntimeError("materials.json 中存在空 audio_url")

    print("materials.json 检查通过。")


def save_replacement_report(replacement_stats):
    report_path = ROOT / "eval" / "user_study" / "data" / "materials_replacement_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(replacement_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print(f"替换记录数量: {len(replacement_stats)}")
    print(f"替换记录已保存: {report_path}")


def main():
    assignments = read_json(ASSIGN_JSON)
    audio_url_map = read_json(AUDIO_URL_JSON)

    method_data = {
        "ours": load_topk_csv(OURS_CSV, "ours"),
        "clap": load_topk_csv(CLAP_CSV, "clap"),
        "clamp3": load_topk_csv(CLAMP3_CSV, "clamp3"),
    }

    attach_audio_urls(method_data, audio_url_map)

    materials, replacement_stats = build_materials(assignments, method_data)
    check_materials(materials)
    save_replacement_report(replacement_stats)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(materials, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("最终 materials.json 已生成:")
    print(OUT_JSON)


if __name__ == "__main__":
    main()