from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

cur_dir = Path(__file__).resolve().parent
src_dir = cur_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from .config import intent_strategy, paths, retrieval_config
    from .core import Emo2Music
except ImportError:
    from application.config import intent_strategy, paths, retrieval_config
    from application.core import Emo2Music


DEFAULT_CASES = [
    {
        "case_id": "case1_high_arousal_negative_soothe",
        "text": "我现在很焦虑，很紧绷，脑子停不下来。",
        "intent": "soothe",
    },
    {
        "case_id": "case2_low_arousal_negative_uplift",
        "text": "我很低落，很丧，什么都不想做。",
        "intent": "uplift",
    },
    {
        "case_id": "case3_resonance_validate",
        "text": "我今晚有点感伤，只想有人懂我。",
        "intent": "validate",
    },
    {
        "case_id": "case4_positive_energize",
        "text": "我今天心情很好，想继续保持活力。",
        "intent": "energize",
    },
]


run_cases_file = ""  # .json/.jsonl path, empty means DEFAULT_CASES

# ============ EXPERIMENT CONFIG ============
# Main experiment: Full26 vs Core10
run_modes = ["full26", "core10"]
# Appendix baseline: off vs Core10
# ===============================================

run_intents: List[str] = []  # e.g. ["soothe", "uplift"], empty means case intent
run_topk = int(retrieval_config["topk"])
run_id = ""  # empty means profile default label
run_out_root = ""  # empty means paths["obs_root"]
run_off_base_mode = "core10"  # real tag_mode when run_modes contains "off"
run_max_cases = 0  # 0 means all cases

PROFILES = {
    "main": {
        "modes": ["full26", "core10"],
        "off_base_mode": "core10",
        "run_id": "full26_vs_core10",
    },
    "appendix": {
        "modes": ["off", "core10"],
        "off_base_mode": "core10",
        "run_id": "off_vs_core10",
    },
}


def load_cases(path: Path | None) -> List[Dict[str, str]]:
    if path is None:
        return [dict(x) for x in DEFAULT_CASES]

    if not path.exists():
        raise FileNotFoundError(f"cases file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("json cases must be a list")
        out = []
        for i, row in enumerate(data):
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            out.append(
                {
                    "case_id": str(row.get("case_id") or f"case_{i + 1}"),
                    "text": text,
                    "intent": str(row.get("intent", "validate")).strip().lower() or "validate",
                }
            )
        return out

    if suffix == ".jsonl":
        out = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            out.append(
                {
                    "case_id": str(row.get("case_id") or f"case_{i + 1}"),
                    "text": text,
                    "intent": str(row.get("intent", "validate")).strip().lower() or "validate",
                }
            )
        return out

    raise ValueError("cases file must be .json or .jsonl")


def _entropy(prob_map: Dict[str, float]) -> float:
    vals = [float(v) for v in prob_map.values() if float(v) > 0.0]
    if not vals:
        return 0.0
    return float(-sum(p * math.log(p + 1e-12) for p in vals))


def _norm_entropy(prob_map: Dict[str, float]) -> float:
    vals = [float(v) for v in prob_map.values() if float(v) > 0.0]
    n = len(vals)
    if n <= 1:
        return 0.0
    ent = -sum(p * math.log(p + 1e-12) for p in vals)
    return float(ent / math.log(n + 1e-12))


def _mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _ensure_unique_dir(base: Path, name: str) -> Path:
    out = base / name
    if not out.exists():
        return out
    idx = 2
    while True:
        cand = base / f"{name}_v{idx}"
        if not cand.exists():
            return cand
        idx += 1


def _build_record(case_id: str, text: str, intent: str, mode: str, result: Dict[str, Any]) -> Dict[str, Any]:
    tau_q = result["f_4"]["tau_selected"]
    tau_sorted = sorted(tau_q.items(), key=lambda x: x[1], reverse=True)
    items = []
    for row in result["f_5"]["results"]:
        items.append(
            {
                "track_id": row.get("song_id"),
                "z_i": row.get("song_va", []),
                "y_i_top_tags": row.get("top_tags", []),
                "mu_d2": float(row.get("mu_d2", 0.0)),
                "va_score": float(row.get("score_va", 0.0)),
                "tag_score": float(row.get("score_tag", 0.0)),
                "total_score": float(row.get("score_total", 0.0)),
                "song_path": row.get("song_path"),
            }
        )

    return {
        "case_id": case_id,
        "mode": mode,
        "input_text": text,
        "regulation_intent": intent,
        "F1": {
            "z_q": result["f_1"].get("z_q", []),
        },
        "F2": {
            "delta_r": result["f_2"].get("delta_va", []),
            "mu_q": result["f_2"].get("mu_q", []),
        },
        "F3": {
            "tag_prototypes": result["f_3"].get("sel_proto", {}),
        },
        "F4": {
            "tag_mode": mode,
            "is_diagnostic_only": mode == "off",
            "tau_q": tau_q,
            "top_tags": [t for t, _ in tau_sorted[:5]],
            "tau_top3_mass": float(sum(w for _, w in tau_sorted[:3])),
            "tau_entropy": _entropy(tau_q),
            "tau_entropy_norm": _norm_entropy(tau_q),
        },
        "F5": {
            "lambda_va": result["f_5"].get("lambda_va"),
            "lambda_tag": result["f_5"].get("lambda_tag"),
            "pool_size": result["f_5"].get("pool_size"),
            "retrieved_items": items,
        },
    }


def _build_topk_rows(case_id: str, mode: str, intent: str, result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for rank, row in enumerate(result["f_5"]["results"], 1):
        rows.append(
            {
                "case_id": case_id,
                "mode": mode,
                "intent": intent,
                "rank": rank,
                "track_id": row.get("song_id"),
                "song_path": row.get("song_path"),
                "va_v": float(row.get("song_va", [0.0, 0.0])[0]),
                "va_a": float(row.get("song_va", [0.0, 0.0])[1]),
                "mu_d2": float(row.get("mu_d2", 0.0)),
                "va_score": float(row.get("score_va", 0.0)),
                "tag_score": float(row.get("score_tag", 0.0)),
                "total_score": float(row.get("score_total", 0.0)),
                "top_tags": json.dumps(row.get("top_tags", []), ensure_ascii=False),
            }
        )
    return rows


def _compute_topk_jaccard(rows: List[Dict[str, Any]], modes: List[str], topk: int = 10) -> Dict[str, float]:
    """Compute mean top-K Jaccard over each case/intent query between two modes."""
    if len(modes) != 2:
        return {}

    m1, m2 = modes[0], modes[1]
    by_query_mode: Dict[Tuple[str, str], Dict[str, set]] = {}
    for row in rows:
        key = (str(row.get("case_id", "")), str(row.get("intent", "")))
        mode = str(row.get("mode", ""))
        rank = int(row.get("rank", 0))
        if mode not in {m1, m2} or rank > topk:
            continue
        by_query_mode.setdefault(key, {m1: set(), m2: set()})
        by_query_mode[key][mode].add(row.get("track_id"))

    jaccards: List[float] = []
    for mode_sets in by_query_mode.values():
        s1, s2 = mode_sets.get(m1, set()), mode_sets.get(m2, set())
        if not s1 or not s2:
            continue
        union = len(s1 | s2)
        if union == 0:
            continue
        jaccards.append(float(len(s1 & s2) / union))

    if not jaccards:
        return {"topk_jaccard": 0.0}
    return {"topk_jaccard": _mean(jaccards)}


def _build_case_studies(records: List[Dict[str, Any]], modes: List[str], max_cases: int = 4) -> List[Dict[str, Any]]:
    if len(modes) != 2:
        return []

    m1, m2 = modes[0], modes[1]
    by_query: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for rec in records:
        key = (str(rec.get("case_id", "")), str(rec.get("regulation_intent", "")))
        by_query.setdefault(key, {})
        by_query[key][str(rec.get("mode", ""))] = rec

    studies = []
    for (case_id, intent), recs in by_query.items():
        if m1 not in recs or m2 not in recs:
            continue

        rec1, rec2 = recs[m1], recs[m2]
        topk1 = rec1["F5"].get("retrieved_items", [])[:run_topk]
        topk2 = rec2["F5"].get("retrieved_items", [])[:run_topk]
        set1 = {x.get("track_id") for x in topk1 if x.get("track_id")}
        set2 = {x.get("track_id") for x in topk2 if x.get("track_id")}
        union = len(set1 | set2)
        jaccard = float(len(set1 & set2) / union) if union > 0 else 0.0

        def tau_top5(rec: Dict[str, Any]) -> List[List[Any]]:
            tau_q = rec["F4"].get("tau_q", {})
            return [[t, float(w)] for t, w in sorted(tau_q.items(), key=lambda x: x[1], reverse=True)[:5]]

        def compact_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out = []
            for rank, it in enumerate(items, 1):
                out.append(
                    {
                        "rank": rank,
                        "track_id": it.get("track_id"),
                        "mu_d2": float(it.get("mu_d2", 0.0)),
                        "top_tags": it.get("y_i_top_tags", []),
                    }
                )
            return out

        shared = sorted(set1 & set2)
        only_m1 = sorted(set1 - set2)
        only_m2 = sorted(set2 - set1)
        anchor = recs.get("core10", rec2)

        studies.append(
            {
                "case_id": case_id,
                "intent": intent,
                "text": anchor.get("input_text", ""),
                "z_q": anchor["F1"].get("z_q", []),
                "mu_q": anchor["F2"].get("mu_q", []),
                "topk_jaccard": jaccard,
                "tau_top5": {
                    m1: tau_top5(rec1),
                    m2: tau_top5(rec2),
                },
                "topk": {
                    m1: compact_items(topk1),
                    m2: compact_items(topk2),
                },
                "shared_track_ids": shared,
                "unique_track_ids": {
                    m1: only_m1,
                    m2: only_m2,
                },
            }
        )

    studies.sort(key=lambda x: (x["topk_jaccard"], x["case_id"], x["intent"]))
    return studies[:max_cases]


def _summary(records: List[Dict[str, Any]], topk_rows: List[Dict[str, Any]], modes: List[str]) -> Dict[str, Any]:
    """
    Summary with ONLY 4 core metrics:
    1. tau_entropy_norm - concentration of tag weights (F4)
    2. tau_top3_mass - top-3 tag mass (F4)
    3. avg_topk_mu_d2 - how well top-K adheres to target mu_q (F5)
    4. topk_jaccard - whether result sets actually differ (F5)
    """
    by_mode: Dict[str, Dict[str, List[float]]] = {}
    
    # Collect F4 metrics from records
    for rec in records:
        mode = rec["mode"]
        by_mode.setdefault(
            mode,
            {
                "tau_entropy_norm": [],
                "tau_top3_mass": [],
            },
        )
        by_mode[mode]["tau_entropy_norm"].append(float(rec["F4"].get("tau_entropy_norm", 0.0)))
        by_mode[mode]["tau_top3_mass"].append(float(rec["F4"]["tau_top3_mass"]))

    # Collect F5 metrics from topk_rows
    for row in topk_rows:
        mode = row["mode"]
        by_mode.setdefault(mode, {"tau_entropy_norm": [], "tau_top3_mass": []})
        by_mode[mode].setdefault("mu_d2", []).append(float(row["mu_d2"]))

    # Compute Jaccard if we have 2 modes
    jaccard_dict = _compute_topk_jaccard(topk_rows, modes, topk=10)
    
    metric_tau_entropy_norm = {mode: _mean(d.get("tau_entropy_norm", [])) for mode, d in by_mode.items()}
    metric_tau_top3_mass = {mode: _mean(d.get("tau_top3_mass", [])) for mode, d in by_mode.items()}
    metric_avg_topk_mu_d2 = {mode: _mean(d.get("mu_d2", [])) for mode, d in by_mode.items()}

    return {
        "tau_entropy_norm": metric_tau_entropy_norm,
        "tau_top3_mass": metric_tau_top3_mass,
        "avg_topk_mu_d2": metric_avg_topk_mu_d2,
        "topk_jaccard": float(jaccard_dict.get("topk_jaccard", 0.0)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused ablation experiments.")
    parser.add_argument(
        "--profile",
        choices=["main", "appendix", "custom"],
        default="main",
        help="main: full26_vs_core10, appendix: off_vs_core10, custom: use module globals.",
    )
    parser.add_argument("--run-id", default="", help="Optional run folder name override.")
    return parser.parse_args()


def _resolve_setup(args: argparse.Namespace) -> Tuple[List[str], str, str]:
    if args.profile in PROFILES:
        profile = PROFILES[args.profile]
        modes = list(profile["modes"])
        off_base_mode = str(profile["off_base_mode"])
        default_run_label = str(profile["run_id"])
    else:
        modes = [str(m).strip() for m in run_modes if str(m).strip()]
        off_base_mode = str(run_off_base_mode)
        default_run_label = "custom_run"

    chosen_run_id = args.run_id.strip() or run_id.strip() or default_run_label
    return modes, off_base_mode, chosen_run_id


def main() -> None:
    args = _parse_args()

    cases_path = Path(run_cases_file) if run_cases_file else None
    cases = load_cases(cases_path)
    if run_max_cases > 0:
        cases = cases[: run_max_cases]
    if not cases:
        raise ValueError("No valid cases to run")

    modes, off_base_mode, chosen_run_id = _resolve_setup(args)
    allowed_modes = {"off", "full26", "core10"}
    bad_modes = [m for m in modes if m not in allowed_modes]
    if bad_modes:
        raise ValueError(f"Unknown mode(s): {bad_modes}")
    if off_base_mode not in {"full26", "core10"}:
        raise ValueError(f"Unknown off base mode: {off_base_mode}")

    intents = [str(x).strip().lower() for x in run_intents if str(x).strip()]
    if intents:
        for x in intents:
            if x not in intent_strategy:
                raise ValueError(f"Unknown intent: {x}")

    run_label = chosen_run_id
    out_root = Path(run_out_root) if run_out_root else Path(paths.get("obs_root", paths["output_json"].parent / "obs_ablation"))
    out_dir = _ensure_unique_dir(out_root, run_label)
    run_label = out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    emo = Emo2Music()
    records: List[Dict[str, Any]] = []
    topk_rows: List[Dict[str, Any]] = []

    old_mode = retrieval_config["tag_mode"]
    old_lambda_tag = float(retrieval_config["lambda_tag"])
    try:
        for case in cases:
            case_id = case["case_id"]
            text = case["text"]
            case_intents = intents or [case.get("intent", "validate")]

            for mode in modes:
                if mode == "off":
                    retrieval_config["tag_mode"] = off_base_mode
                    retrieval_config["lambda_tag"] = 0.0
                else:
                    retrieval_config["tag_mode"] = mode
                    retrieval_config["lambda_tag"] = old_lambda_tag

                for intent in case_intents:
                    result = emo.run(text=text, intent=intent)
                    rec = _build_record(case_id, text, intent, mode, result)
                    records.append(rec)
                    topk_rows.extend(_build_topk_rows(case_id, mode, intent, result))

                    print(
                        f"[done] case={case_id} | mode={mode} | intent={intent} | "
                        f"pool={result['f_5']['pool_size']} | top1={result['f_5']['results'][0]['score_total']:.5f}"
                    )
    finally:
        retrieval_config["tag_mode"] = old_mode
        retrieval_config["lambda_tag"] = old_lambda_tag

    records_jsonl = out_dir / "records.jsonl"
    with records_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    topk_csv = out_dir / "topk_rows.csv"
    with topk_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "case_id",
            "mode",
            "intent",
            "rank",
            "track_id",
            "song_path",
            "va_v",
            "va_a",
            "mu_d2",
            "top_tags",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in topk_rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    summary = {
        "run_id": run_label,
        "out_dir": str(out_dir),
        "topk": int(run_topk),
        "profile": args.profile,
        "modes": modes,
        "intents": intents or "case_intent",
        "cfg": {
            "q_key": retrieval_config["q_key"],
            "alpha": retrieval_config["alpha"],
            "score_mode": retrieval_config.get("score_mode", "va_constrained_rerank"),
            "lambda_va": retrieval_config["lambda_va"],
            "lambda_tag": old_lambda_tag,
            "va_prerank_mult": retrieval_config.get("va_prerank_mult"),
            "va_prerank_min": retrieval_config.get("va_prerank_min"),
        },
        "stats": _summary(records, topk_rows, modes),
    }

    case_studies = _build_case_studies(records, modes, max_cases=4)
    case_studies_path = out_dir / "case_studies.json"
    case_studies_path.write_text(json.dumps(case_studies, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Saved ===")
    print(f"records: {records_jsonl}")
    print(f"topk   : {topk_csv}")
    print(f"summary: {summary_path}")
    print(f"cases  : {case_studies_path}")


if __name__ == "__main__":
    main()


