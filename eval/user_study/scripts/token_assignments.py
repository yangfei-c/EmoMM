# -*- coding: utf-8 -*-

import json
import random
import string
from collections import Counter
from pathlib import Path

# =========================
# 直接改这里
# =========================
N_PARTICIPANTS = 100
TRIALS_PER_USER = 3
SEED = 20260514

BASE_URL = "http://127.0.0.1:8000/index.html"
# 部署后改成正式地址，例如：
# BASE_URL = "https://你的前端域名/index.html"

# =========================
# 路径
# 当前文件位置：
# eval/user_study/scripts/token_assignments.py
# parents[3] = 项目根目录 EmoMusicRec
# =========================
ROOT = Path(__file__).resolve().parents[3]

QUERY_JSON = ROOT / "eval" / "queries_zh16.json"

OUT_DIR = ROOT / "eval" / "user_study" / "backend" / "data"
OUT_JSON = OUT_DIR / "token_assignments.json"
OUT_LINKS = OUT_DIR / "user_links.txt"
OUT_SUMMARY = OUT_DIR / "assignment_summary.json"

METHOD_ORDERS = [
    ["ours", "clap", "clamp3"],
    ["ours", "clamp3", "clap"],
    ["clap", "ours", "clamp3"],
    ["clap", "clamp3", "ours"],
    ["clamp3", "ours", "clap"],
    ["clamp3", "clap", "ours"],
]


def make_token(i, rng):
    chars = string.ascii_uppercase + string.digits
    return f"P{i:03d}_" + "".join(rng.choice(chars) for _ in range(8))


def load_queries():
    queries = json.loads(QUERY_JSON.read_text(encoding="utf-8"))
    allowed = {
        "maintain": "保持当前状态",
        "calm": "平静下来",
        "comfort": "获得安慰",
        "energize": "振作起来",
    }

    for q in queries:
        sid = q["strategy"]
        if sid not in allowed:
            raise ValueError(f"{q['query_id']} strategy 不合法: {sid}")
        if q["strategy_zh"] != allowed[sid]:
            raise ValueError(f"{q['query_id']} strategy_zh 不固定: {q['strategy_zh']}")

    return queries


def build_query_slots(queries, total_trials, rng):
    qids = [q["query_id"] for q in queries]
    slots = [qids[i % len(qids)] for i in range(total_trials)]
    rng.shuffle(slots)
    return slots


def pick_queries(slots, k):
    picked = []
    rest = []

    while slots and len(picked) < k:
        qid = slots.pop()
        if qid not in picked:
            picked.append(qid)
        else:
            rest.append(qid)

    slots[:0] = rest

    if len(picked) < k:
        raise RuntimeError("query slot 不够，无法给用户分配不重复 query")

    return picked


def main():
    rng = random.Random(SEED)
    queries = load_queries()
    qmap = {q["query_id"]: q for q in queries}

    total_trials = N_PARTICIPANTS * TRIALS_PER_USER
    slots = build_query_slots(queries, total_trials, rng)

    users = []
    all_trials = []

    for p_idx in range(1, N_PARTICIPANTS + 1):
        token = make_token(p_idx, rng)
        url = f"{BASE_URL}?token={token}" if BASE_URL else token
        picked = pick_queries(slots, TRIALS_PER_USER)

        user = {
            "token": token,
            "participant_index": p_idx,
            "url": url,
            "trials": [],
        }

        for t_idx, qid in enumerate(picked, start=1):
            q = qmap[qid]
            order = METHOD_ORDERS[(len(all_trials)) % len(METHOD_ORDERS)]

            trial = {
                "trial_index": t_idx,
                "query_id": qid,
                "strategy": q["strategy"],
                "query_text": q["text"],
                "method_A": order[0],
                "method_B": order[1],
                "method_C": order[2],
            }

            user["trials"].append(trial)
            all_trials.append({
                "token": token,
                "participant_index": p_idx,
                **trial,
            })

        users.append(user)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_JSON.write_text(
        json.dumps(users, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    OUT_LINKS.write_text(
        "\n".join(u["url"] for u in users),
        encoding="utf-8",
    )

    summary = {
        "n_participants": N_PARTICIPANTS,
        "trials_per_user": TRIALS_PER_USER,
        "total_trials": len(all_trials),
        "query_counts": dict(sorted(Counter(x["query_id"] for x in all_trials).items())),
        "strategy_counts": dict(sorted(Counter(x["strategy"] for x in all_trials).items())),
        "method_A_counts": dict(sorted(Counter(x["method_A"] for x in all_trials).items())),
        "method_B_counts": dict(sorted(Counter(x["method_B"] for x in all_trials).items())),
        "method_C_counts": dict(sorted(Counter(x["method_C"] for x in all_trials).items())),
    }

    OUT_SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("完成 token 分配")
    print(f"JSON: {OUT_JSON}")
    print(f"Links: {OUT_LINKS}")
    print(f"Summary: {OUT_SUMMARY}")

    print("\nquery 出现次数：")
    for k, v in summary["query_counts"].items():
        print(k, v)

    print("\nstrategy 出现次数：")
    for k, v in summary["strategy_counts"].items():
        print(k, v)


if __name__ == "__main__":
    main()