import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"/")
sys.path.insert(0, str(ROOT / "src"))

from core.music_repository import load_tracks

OUT_JSON = ROOT / "data" / "eval" / "candidate_pool.json"

SEED = 2026
MAX_N = 1000
PER_QUADRANT = 250


def quadrant(va):
    v, a = float(va[0]), float(va[1])
    if v >= 0 and a >= 0:
        return 0
    if v >= 0 and a < 0:
        return 1
    if v < 0 and a >= 0:
        return 2
    return 3


def pick_by_tags(items, n):
    tag_buckets = defaultdict(list)

    for x in items:
        for tag in x["display_tags"]:
            tag_buckets[tag].append(x)

    for tag in tag_buckets:
        random.shuffle(tag_buckets[tag])

    tags = list(tag_buckets.keys())
    random.shuffle(tags)

    picked = []
    used = set()

    while len(picked) < n:
        changed = False

        for tag in tags:
            while tag_buckets[tag]:
                x = tag_buckets[tag].pop()

                if x["song_id"] in used:
                    continue

                picked.append(x)
                used.add(x["song_id"])
                changed = True
                break

            if len(picked) >= n:
                break

        if not changed:
            break

    if len(picked) < n:
        random.shuffle(items)
        for x in items:
            if x["song_id"] not in used:
                picked.append(x)
                used.add(x["song_id"])
            if len(picked) >= n:
                break

    return picked


def main():
    random.seed(SEED)

    groups = defaultdict(list)

    for t in load_tracks():
        p = Path(str(t["song_path"]))

        if not p.exists():
            continue

        tags = t.get("display_tags", [])
        va = t.get("va", [])

        if not tags or len(va) < 2:
            continue

        item = {
            "song_id": str(t["song_id"]),
            "song_path": str(p),
            "display_tags": [str(x) for x in tags],
            "va": [float(va[0]), float(va[1])],
        }

        groups[quadrant(item["va"])].append(item)

    selected = []
    used = set()

    for q in range(4):
        part = pick_by_tags(groups[q], PER_QUADRANT)
        selected.extend(part)
        used.update(x["song_id"] for x in part)
        print(f"quadrant {q}: {len(part)}")

    if len(selected) < MAX_N:
        rest = []

        for q in range(4):
            for x in groups[q]:
                if x["song_id"] not in used:
                    rest.append(x)

        random.shuffle(rest)
        selected.extend(rest[:MAX_N - len(selected)])

    selected = selected[:MAX_N]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"完成: {OUT_JSON}")
    print(f"数量: {len(selected)}")


if __name__ == "__main__":
    main()