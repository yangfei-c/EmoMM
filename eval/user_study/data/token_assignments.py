import csv
import json
import random
import string
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

QUERY_JSON = ROOT / "queries_zh16.json"

OUT_DIR = ROOT / "eval" / "user_study" / "data"
OUT_CSV = OUT_DIR / "token_assignments.csv"
OUT_JSON = OUT_DIR / "token_assignments.json"
OUT_LINKS = OUT_DIR / "user_links.txt"

n_tokens = 40
trials_per_token = 3
seed = 2026
base_url = "http://127.0.0.1:8000/index.html"  # 以后有网页地址再填，比如 http://127.0.0.1:8000/index.html

method_orders = [
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


def main():
    rng = random.Random(seed)
    queries = json.loads(QUERY_JSON.read_text(encoding="utf-8"))
    query_ids = [q["query_id"] for q in queries]
    query_map = {q["query_id"]: q for q in queries}

    total_trials = n_tokens * trials_per_token
    slots = []

    for i in range(total_trials):
        slots.append(query_ids[i % len(query_ids)])

    rng.shuffle(slots)

    rows = []

    for p_idx in range(1, n_tokens + 1):
        token = make_token(p_idx, rng)

        # 保证同一个人 3 个 query 不重复
        picked = []
        while len(picked) < trials_per_token:
            qid = slots.pop()
            if qid not in picked:
                picked.append(qid)
            else:
                slots.insert(0, qid)

        for t_idx, qid in enumerate(picked, start=1):
            q = query_map[qid]
            order = method_orders[(p_idx + t_idx - 2) % len(method_orders)]
            url = f"{base_url}?token={token}" if base_url else ""

            rows.append({
                "token": token,
                "participant_index": p_idx,
                "trial_index": t_idx,
                "query_id": qid,
                "strategy": q["strategy"],
                "query_text": q["text"],
                "method_A": order[0],
                "method_B": order[1],
                "method_C": order[2],
                "url": url,
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for r in rows:
        grouped.setdefault(r["token"], {
            "token": r["token"],
            "participant_index": r["participant_index"],
            "url": r["url"],
            "trials": [],
        })
        grouped[r["token"]]["trials"].append({
            "trial_index": r["trial_index"],
            "query_id": r["query_id"],
            "strategy": r["strategy"],
            "query_text": r["query_text"],
            "method_A": r["method_A"],
            "method_B": r["method_B"],
            "method_C": r["method_C"],
        })

    OUT_JSON.write_text(
        json.dumps(list(grouped.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    OUT_LINKS.write_text(
        "\n".join(x["url"] if x["url"] else x["token"] for x in grouped.values()),
        encoding="utf-8",
    )

    print("完成 token 分配")
    print(f"CSV: {OUT_CSV}")
    print(f"JSON: {OUT_JSON}")
    print(f"Links: {OUT_LINKS}")

    print("\nquery 出现次数：")
    for qid, n in sorted(Counter(r["query_id"] for r in rows).items()):
        print(qid, n)


if __name__ == "__main__":
    main()