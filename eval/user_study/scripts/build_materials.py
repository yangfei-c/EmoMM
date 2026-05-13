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

# 新版正式问卷：每个方法只展示 1 首音乐
TOP_K_SHOW = 1

# 仍然从 top5.csv 里读取前 5 首作为候选。
# 如果同一个 trial 内 A/B/C 撞歌，可以用 rank2/rank3 兜底替换。
CANDIDATE_K = 5

# 正式评估建议 False：
# - False：只避免同一个 trial 内 A/B/C 撞同一首歌；
# - True：同一个 participant 的 3 个 trial 内都不重复 song_id。
#
# 现在建议 False，因为如果 CLAP 本身 rank1 重复很多，
# 那也是 baseline 的真实表现，不应该用替换把它隐藏掉。
DEDUP_WITHIN_PARTICIPANT = False

# 新版问卷只保留一个主评分维度
RATING_FIELDS = ["regulation_suitability"]

RATING_PROMPTS = {
    "regulation_suitability": "这首音乐适合帮助我从当前情绪调节到目标状态。"
}

PREFERENCE_PROMPT = "本轮三首音乐中，我最愿意选择哪一首来帮助自己完成情绪调节？"


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


def clean_text(x):
    return str(x or "").strip()


def load_topk_csv(csv_path, method_name):
    """
    读取某个方法的 top5.csv，保留 rank <= CANDIDATE_K。

    返回:
    method_data:
    {
        "q01": [rank1, rank2, ..., rank5],
        ...
    }

    query_meta:
    {
        "q01": {
            "text": "...",
            "current_state": "...",
            "regulation_goal": "...",
            "strategy": "...",
            "strategy_zh": "...",
            "strength": ...
        },
        ...
    }
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    data = {}
    query_meta = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {"query_id", "rank", "song_id", "text", "strategy"}
        missing_cols = required_cols - set(reader.fieldnames or [])

        if missing_cols:
            raise RuntimeError(f"{csv_path} 缺少必要列: {missing_cols}")

        for row in reader:
            qid = clean_text(row["query_id"])
            rank = int(row["rank"])
            song_id = clean_text(row["song_id"])

            if not qid:
                raise ValueError(f"{csv_path} 存在空 query_id")

            if not song_id:
                raise ValueError(f"{csv_path} {qid} rank={rank} 存在空 song_id")

            # 记录 query 级别元信息
            if qid not in query_meta:
                text = clean_text(row.get("text"))
                current_state = clean_text(row.get("current_state"))
                regulation_goal = clean_text(row.get("regulation_goal"))
                strategy = clean_text(row.get("strategy"))
                strategy_zh = clean_text(row.get("strategy_zh")) or strategy
                strength = row.get("strength", "")

                if not current_state:
                    # 兼容旧 CSV，但新版正式实验应该有 current_state
                    current_state = text

                if not text:
                    # 兼容缺 text 的情况
                    text = f"{current_state}{regulation_goal}".strip()

                query_meta[qid] = {
                    "query_id": qid,
                    "text": text,
                    "current_state": current_state,
                    "regulation_goal": regulation_goal,
                    "strategy": strategy,
                    "strategy_zh": strategy_zh,
                    "strength": strength,
                }

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

            # Ours 可能有 z_q / mu_q；保留到 item 里方便排查，但前端不显示
            if "z_q" in row:
                item["z_q"] = row.get("z_q", "")

            if "mu_q" in row:
                item["mu_q"] = row.get("mu_q", "")

            data.setdefault(qid, []).append(item)

    for qid in data:
        data[qid] = sorted(data[qid], key=lambda x: x["rank"])

    print(
        f"{method_name}: query 数量={len(data)}, "
        f"候选 item 数量={sum(len(v) for v in data.values())}"
    )

    return data, query_meta


def merge_query_meta(*meta_maps):
    """
    合并三个方法 CSV 中的 query meta。
    理论上三份 CSV 的 query meta 应一致。
    这里优先使用第一个非空版本。
    """
    merged = {}

    for meta in meta_maps:
        for qid, info in meta.items():
            if qid not in merged:
                merged[qid] = deepcopy(info)
                continue

            # 做简单一致性检查，不中断，只打印提示
            old = merged[qid]
            for key in ["text", "current_state", "regulation_goal", "strategy", "strategy_zh"]:
                if clean_text(old.get(key)) and clean_text(info.get(key)):
                    if clean_text(old.get(key)) != clean_text(info.get(key)):
                        print(
                            f"[Meta Warning] {qid} 字段 {key} 在不同 CSV 中不一致。"
                            f" 已使用先读到的版本。"
                        )

    return merged


def attach_audio_urls(method_data, audio_url_map):
    """
    给所有候选 item 写入 Spaces audio_url。
    这里包括 rank2-rank5，所以 audio_clip_urls.json 需要包含 top5 里可能用到的候选。
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
            f"请先运行 prepare_audio_clips.py 补齐这些歌曲的音频裁剪和上传。"
        )


def select_items_without_duplicate(candidates, used_song_ids, top_k=TOP_K_SHOW):
    """
    从候选列表中选 top_k 个。

    新版 top_k=1：
    - 优先选没有在 used_song_ids 中出现过的 song_id；
    - 如果不够，再允许重复兜底；
    - 主要用于避免同一个 trial 内 A/B/C 撞同一首歌。
    """
    selected = []
    selected_ids = set()

    # 第一轮：优先选未使用过的，且当前组内不重复的
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

    # 第二轮兜底：如果不够，允许用已经出现过的，但当前组内仍不重复
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

    if key not in method_data:
        raise KeyError(f"method_data 缺少方法: {key}")

    if qid not in method_data[key]:
        raise KeyError(f"{key} 缺少 query_id={qid}")

    candidates = method_data[key][qid]

    if len(candidates) < TOP_K_SHOW:
        raise RuntimeError(
            f"{key} 的 {qid} 只有 {len(candidates)} 个候选，不足 Top-{TOP_K_SHOW}"
        )

    return candidates


def get_query_meta(qid, trial, query_meta):
    """
    优先使用新版 top5 CSV 里的 query meta。
    如果没有，则兜底使用 token_assignments 里的旧字段。
    """
    if qid in query_meta:
        return query_meta[qid]

    print(f"[Meta Warning] query_meta 中找不到 {qid}，使用 token_assignments 中的兜底字段。")

    text = clean_text(trial.get("query_text"))
    strategy = clean_text(trial.get("strategy"))

    return {
        "query_id": qid,
        "text": text,
        "current_state": text,
        "regulation_goal": "",
        "strategy": strategy,
        "strategy_zh": strategy,
        "strength": "",
    }


def build_materials(assignments, method_data, query_meta):
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
            qid = clean_text(trial["query_id"])
            q_meta = get_query_meta(qid, trial, query_meta)

            # 正式实验建议只在同一个 trial 内避免 A/B/C 撞歌。
            if DEDUP_WITHIN_PARTICIPANT:
                used_for_this_trial = participant_used_song_ids
            else:
                used_for_this_trial = set()

            arms = {}

            for arm_label in ["A", "B", "C"]:
                method_field = f"method_{arm_label}"
                method_name = trial[method_field]
                method_key = normalize_method_name(method_name)

                candidates = get_candidates(method_name, qid, method_data)

                original_top1 = [x["song_id"] for x in candidates[:TOP_K_SHOW]]

                selected = select_items_without_duplicate(
                    candidates=candidates,
                    used_song_ids=used_for_this_trial,
                    top_k=TOP_K_SHOW,
                )

                selected_ids = [x["song_id"] for x in selected]

                if selected_ids != original_top1:
                    replacement_stats.append({
                        "token": user["token"],
                        "participant_index": user["participant_index"],
                        "trial_index": trial["trial_index"],
                        "query_id": qid,
                        "arm": arm_label,
                        "method": method_key,
                        "original_top1": original_top1,
                        "selected": selected_ids,
                    })

                arms[arm_label] = {
                    # 前端不显示 method；后续合并分析需要保留
                    "method": method_key,
                    "items": selected,
                }

                if DEDUP_WITHIN_PARTICIPANT:
                    # 如果是 participant 级去重，selected 里的 song_id 已经加入 participant_used_song_ids
                    participant_used_song_ids.update(selected_ids)

            trial_item = {
                "trial_index": trial["trial_index"],
                "query_id": qid,

                # 新版 query 展示字段
                "query_text": q_meta.get("text", ""),
                "current_state": q_meta.get("current_state", ""),
                "regulation_goal": q_meta.get("regulation_goal", ""),
                "strategy": q_meta.get("strategy", trial.get("strategy", "")),
                "strategy_zh": q_meta.get("strategy_zh", q_meta.get("strategy", "")),
                "strength": q_meta.get("strength", ""),

                # 前端只显示 A/B/C，不显示 method
                "A": arms["A"],
                "B": arms["B"],
                "C": arms["C"],

                # 新版问卷只保留一个主评分维度
                "rating_fields": RATING_FIELDS,
                "rating_prompts": RATING_PROMPTS,
                "preference_prompt": PREFERENCE_PROMPT,

                "top_k": TOP_K_SHOW,
            }

            user_item["trials"].append(trial_item)

        materials.append(user_item)

    return materials, replacement_stats


def check_materials(materials):
    empty_url = 0
    item_count = 0
    trial_count = 0
    within_trial_duplicate_count = 0
    within_participant_duplicate_count = 0
    missing_query_meta_count = 0

    for user in materials:
        participant_seen = set()

        for trial in user["trials"]:
            trial_count += 1
            trial_seen = set()

            if not trial.get("current_state"):
                missing_query_meta_count += 1

            if not trial.get("regulation_goal"):
                missing_query_meta_count += 1

            if not trial.get("strategy_zh"):
                missing_query_meta_count += 1

            if trial.get("rating_fields") != RATING_FIELDS:
                raise RuntimeError(
                    f"{user['token']} {trial['query_id']} rating_fields 不符合新版设置"
                )

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

                    if sid in trial_seen:
                        within_trial_duplicate_count += 1

                    if sid in participant_seen:
                        within_participant_duplicate_count += 1

                    trial_seen.add(sid)
                    participant_seen.add(sid)

    print("=" * 80)
    print(f"用户数量: {len(materials)}")
    print(f"trial 数量: {trial_count}")
    print(f"音频 item 引用数量: {item_count}")
    print(f"空 audio_url 数量: {empty_url}")
    print(f"同一 trial 内重复 song_id 数量: {within_trial_duplicate_count}")
    print(f"同一 participant 内重复 song_id 数量: {within_participant_duplicate_count}")
    print(f"query 展示字段缺失计数: {missing_query_meta_count}")

    if empty_url > 0:
        raise RuntimeError("materials.json 中存在空 audio_url")

    if within_trial_duplicate_count > 0:
        raise RuntimeError("同一个 trial 内 A/B/C 出现重复 song_id，请检查候选或去重逻辑")

    if missing_query_meta_count > 0:
        raise RuntimeError("materials.json 中存在 current_state / regulation_goal / strategy_zh 缺失")

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


def save_materials_preview(materials):
    """
    保存一个简短预览，方便人工检查每个 trial 的 A/B/C 是哪首歌。
    """
    preview_path = ROOT / "eval" / "user_study" / "data" / "materials_preview.csv"
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for user in materials:
        for trial in user["trials"]:
            for arm in ["A", "B", "C"]:
                method = trial[arm]["method"]
                item = trial[arm]["items"][0]

                rows.append({
                    "token": user["token"],
                    "participant_index": user["participant_index"],
                    "trial_index": trial["trial_index"],
                    "query_id": trial["query_id"],
                    "strategy": trial["strategy"],
                    "strategy_zh": trial["strategy_zh"],
                    "arm": arm,
                    "method": method,
                    "song_id": item["song_id"],
                    "rank": item["rank"],
                    "score": item.get("score", ""),
                    "audio_url": item.get("audio_url", ""),
                    "current_state": trial["current_state"],
                    "regulation_goal": trial["regulation_goal"],
                })

    with preview_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "token",
                "participant_index",
                "trial_index",
                "query_id",
                "strategy",
                "strategy_zh",
                "arm",
                "method",
                "song_id",
                "rank",
                "score",
                "audio_url",
                "current_state",
                "regulation_goal",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print(f"materials 预览 CSV 已保存: {preview_path}")
    print(f"预览行数: {len(rows)}")


def main():
    assignments = read_json(ASSIGN_JSON)
    audio_url_map = read_json(AUDIO_URL_JSON)

    ours_data, ours_meta = load_topk_csv(OURS_CSV, "ours")
    clap_data, clap_meta = load_topk_csv(CLAP_CSV, "clap")
    clamp3_data, clamp3_meta = load_topk_csv(CLAMP3_CSV, "clamp3")

    method_data = {
        "ours": ours_data,
        "clap": clap_data,
        "clamp3": clamp3_data,
    }

    query_meta = merge_query_meta(
        ours_meta,
        clap_meta,
        clamp3_meta,
    )

    print("=" * 80)
    print(f"合并后 query_meta 数量: {len(query_meta)}")

    attach_audio_urls(method_data, audio_url_map)

    materials, replacement_stats = build_materials(
        assignments=assignments,
        method_data=method_data,
        query_meta=query_meta,
    )

    check_materials(materials)
    save_replacement_report(replacement_stats)
    save_materials_preview(materials)

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