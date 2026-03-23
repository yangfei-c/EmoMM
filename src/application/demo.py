from __future__ import annotations

import sys
from pathlib import Path

cur_dir = Path(__file__).resolve().parent
src_dir = cur_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from .config import retrieval_config
    from .core import Emo2Music
except ImportError:
    from application.config import retrieval_config
    from application.core import Emo2Music

topk = int(retrieval_config["topk"])
VALID_INTENTS = {"validate", "soothe", "uplift", "energize"}


def main() -> None:
    text = input("Please enter your current feeling text: ").strip()
    intent = input("Please enter your intent(validate/soothe/uplift/energize): ").strip().lower()
    if not intent:
        intent = "validate"
    if intent not in VALID_INTENTS:
        print(f"Unknown intent '{intent}', fallback to 'validate'.")
        intent = "validate"

    emo2music = Emo2Music()
    result = emo2music.run(text=text, intent=intent, topk=topk)
    save_path = emo2music.save_result(result)

    z_q = result["f_1"]["z_q"]
    mu_q = result["f_2"]["mu_q"]
    delta = result["f_2"]["delta_va"]
    sel_proto = result["f_3"]["sel_proto"]
    tau_sorted = sorted(result["f_4"]["tau_selected"].items(), key=lambda x: x[1], reverse=True)
    d2_map = result["f_4"]["distance_squared_to_mu"]

    tau_top = " | ".join([f"{t}:{w:.3f}(d2={d2_map.get(t, 0.0):.3f})" for t, w in tau_sorted[:5]])
    proto_top = ", ".join([f"{t}:{[round(v, 3) for v in va]}" for t, va in list(sel_proto.items())[:6]])

    print("\n=== Compact View ===")
    print(
        f"intent={result['meta']['intent']} | mode={result['meta']['tag_mode']} | "
        f"topk={result['meta']['topk']} | pool={result['meta']['pool_size']}"
    )
    print(f"z_q={[round(x, 4) for x in z_q]} | delta={[round(x, 4) for x in delta]} | mu_q={[round(x, 4) for x in mu_q]}")
    print(f"proto(sample) {proto_top}")
    print(f"tau(top5) {tau_top}")

    print("\n=== Rank (compact) ===")
    print("rk | total    | va_norm  | tag_norm | mu_d2    | va               | top_tags")
    for i, item in enumerate(result["f_5"]["results"], 1):
        top_tags = ",".join([f"{t}:{v:.2f}" for t, v in item.get("top_tags", [])[:3]])
        print(
            f"{i:>2d} | "
            f"{item['score_total']:>8.5f} | "
            f"{item['score_va']:>8.5f} | "
            f"{item['score_tag']:>8.5f} | "
            f"{item.get('mu_d2', 0.0):>8.5f} | "
            f"{str([round(x, 3) for x in item['song_va']]):<16} | "
            f"{top_tags}"
        )

    print("\n=== Paths ===")
    for i, item in enumerate(result["f_5"]["results"], 1):
        print(f"{i:>2d}. {item.get('song_path')}")

    print(f"\nSaved result: {Path(save_path)}")


if __name__ == "__main__":
    main()

