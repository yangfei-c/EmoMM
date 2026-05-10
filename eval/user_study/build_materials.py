# -*- coding: utf-8 -*-

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ASSIGN_JSON = ROOT / "eval" / "user_study" / "data" / "token_assignments.json"
OUT_JSON = ROOT / "eval" / "user_study" / "data" / "materials.json"

TOP_K = 3


def empty_recs(method):
    return {
        "method": method,
        "items": []
    }


def main():
    assignments = json.loads(ASSIGN_JSON.read_text(encoding="utf-8"))

    materials = []

    for user in assignments:
        user_item = {
            "token": user["token"],
            "participant_index": user["participant_index"],
            "trials": []
        }

        for trial in user["trials"]:
            trial_item = {
                "trial_index": trial["trial_index"],
                "query_id": trial["query_id"],
                "strategy": trial["strategy"],
                "query_text": trial["query_text"],

                "A": empty_recs(trial["method_A"]),
                "B": empty_recs(trial["method_B"]),
                "C": empty_recs(trial["method_C"]),

                "rating_fields": [
                    "regulation_fit",
                    "helpfulness",
                    "satisfaction"
                ],
                "top_k": TOP_K
            }

            user_item["trials"].append(trial_item)

        materials.append(user_item)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(materials, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("完成 materials 空壳生成")
    print(f"输出: {OUT_JSON}")
    print(f"用户数: {len(materials)}")
    print(f"trial 数: {sum(len(x['trials']) for x in materials)}")


if __name__ == "__main__":
    main()