from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

CUR_DIR = Path(__file__).resolve().parent
SRC_DIR = CUR_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text_va.inference import load_model_and_tokenizer, predict_emotion
try:
    from .config import (
        intent_strategy, paths, retrieval_config, tag_choices, tag2idx, tags_26, text_config,
    )
except ImportError:
    from application.config import (
        intent_strategy, paths, retrieval_config, tag_choices, tag2idx, tags_26, text_config,
    )


class Emo2Music:
    def __init__(self) -> None:
        self.rows = self._load_json(paths["semantic_db"])
        self.proto_data = self._load_prototypes()
        self.sep_data = self._load_separation()
        self._model_pack = None

    @staticmethod
    def _load_json(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prototypes(self) -> Dict:
        path = paths["prototype_json"]
        if path.exists():
            return self._load_json(path)
        return self._build_prototypes(self.rows)

    def _load_separation(self) -> Dict:
        path = paths["separation_json"]
        if path.exists():
            return self._load_json(path)
        return {}

    # 确保 valence/arousal 范围为 [-1, 1]
    @staticmethod
    def _clip_m11(x: float) -> float:
        return float(max(-1.0, min(1.0, x)))

    # 确保 [v, a] 范围为 [-1, 1]
    @classmethod
    def _clip2(cls, z: List[float]) -> List[float]:
        return [cls._clip_m11(float(z[0])), cls._clip_m11(float(z[1]))]

    @staticmethod
    def _norm_intent(intent: str) -> str:
        key = intent.strip().lower()
        return key or "validate"


    def _load_model(self):
        if self._model_pack is not None:
            return self._model_pack
        ckpt = paths["text_ckpt"]
        if not ckpt.exists():
            raise FileNotFoundError(f"Text VA checkpoint not found: {ckpt}")
        self._model_pack = load_model_and_tokenizer(str(ckpt), model_name=text_config["model_name"])
        return self._model_pack

    # Formula 1,z_q=f(text)
    def infer_z(self, text: str) -> List[float]:
        model, tokenizer, device = self._load_model()
        v, a = predict_emotion(
            text,
            model,
            tokenizer,
            device,
            max_length=int(text_config["max_length"]),
        )
        return self._clip2([v, a])

    # Formula 2,mu_q = z_q + delta(intent, z_q)
    def infer_mu(self, z_q: List[float], intent: str) -> Tuple[List[float], List[float]]:
        v, a = float(z_q[0]), float(z_q[1])
        key = self._norm_intent(intent)
        if key not in intent_strategy:
            raise ValueError(f"Unknown intent: {intent}")

        cfg = intent_strategy[key]
        if cfg["type"] == "zero":
            dv, da = 0.0, 0.0
        elif cfg["type"] == "delta":
            dv = cfg.get("dv_pos", 0.0) * (1.0 - v)
            if "da_neg" in cfg:
                da = -cfg["da_neg"] * (a + 1.0)
            else:
                da = cfg.get("da_pos", 0.0) * (1.0 - a)
        else:
            dv = cfg["kv"] * (cfg["target_v"] - v)
            da = cfg["ka"] * (cfg["target_a"] - a)

        delta = [float(dv), float(da)]
        mu_q = self._clip2([v + dv, a + da])
        return mu_q, delta

    # Formula 3
    def _build_prototypes(self, rows: List[Dict]) -> Dict:
        q_key = str(retrieval_config["q_key"])
        q = float(q_key)#0.95
        out = {"q_list": [q], "main_q": q, "tags": {}}

        vecs = [r.get("semantic_vector_28d") for r in rows]
        valid = [v for v in vecs if v is not None and len(v) >= 28]
        va = np.asarray([[float(v[0]), float(v[1])] for v in valid], dtype=np.float64)

        for tag in tags_26:#26个标签一个一个处理，如piano
            idx = 2 + tag2idx[tag]
            score = np.asarray([float(v[idx]) for v in valid], dtype=np.float64)#所有piano得分
            thr = float(np.quantile(score, q))#95分位数作为阈值
            mask = score >= thr#判断score是否大于等于阈值，得到布尔数组
            weight = score[mask] - thr + 1e-12
            proto = (va[mask] * weight[:, None]).sum(axis=0) / weight.sum()
            out["tags"][tag] = {
                "tag_index_26": tag2idx[tag],
                "prototypes": {
                    q_key: {
                        "prototype_va": [float(proto[0]), float(proto[1])],
                    }
                },
            }
        return out

    def _selected_tags(self) -> List[str]:
        mode = str(retrieval_config["tag_mode"]).strip()
        ordered = [t for t, _ in sorted(tag2idx.items(), key=lambda x: x[1])]

        if mode == "full26":
            return ordered
        if mode in self.sep_data and isinstance(self.sep_data[mode], list):
            keep = [str(x) for x in self.sep_data[mode] if str(x) in tag2idx]
        else:
            keep = [t for t in tag_choices.get(mode, []) if t in tag2idx]

        if not keep:
            raise ValueError(f"Empty tag subset for mode={mode}")
        keep_set = set(keep)
        return [t for t in ordered if t in keep_set]

    # Formula 4
    def infer_tau(self, mu_q: List[float]) -> Tuple[List[float], Dict[str, float], Dict[str, float]]:
        q_key = str(retrieval_config["q_key"])
        alpha = float(retrieval_config["alpha"])
        tags = self._selected_tags()

        energies = []
        distances = {}
        for tag in tags:
            c = self.proto_data["tags"][tag]["prototypes"][q_key]["prototype_va"]
            d2 = (mu_q[0] - float(c[0])) ** 2 + (mu_q[1] - float(c[1])) ** 2
            distances[tag] = float(d2)
            energies.append(-alpha * d2)

        m = max(energies)
        exps = [math.exp(e - m) for e in energies]
        s = sum(exps) + 1e-12
        tau_selected = {tag: float(v / s) for tag, v in zip(tags, exps)}

        tau_full26 = [0.0] * 26
        for tag, w in tau_selected.items():
            tau_full26[tag2idx[tag]] = float(w)

        return tau_full26, tau_selected, distances

    def get_all_proto(self) -> Dict[str, List[float]]:
        q_key = str(retrieval_config["q_key"])
        return {
            tag: self.proto_data["tags"][tag]["prototypes"][q_key]["prototype_va"]
            for tag in tags_26
        }

    def get_sel_proto(self) -> Dict[str, List[float]]:
        q_key = str(retrieval_config["q_key"])
        tags = self._selected_tags()
        return {
            tag: self.proto_data["tags"][tag]["prototypes"][q_key]["prototype_va"]
            for tag in tags
        }

    @staticmethod
    def _rank_to_unit(scores: List[float], descending: bool) -> List[float]:
        n = len(scores)
        if n == 0:
            return []
        if n == 1:
            return [1.0]
        order = sorted(range(n), key=lambda i: scores[i], reverse=descending)
        out = [0.0] * n
        denom = max(1, n - 1)
        for rank, idx in enumerate(order):
            out[idx] = 1.0 - rank / denom
        return out

    def _collect_candidates(self, mu_q: List[float], tau_full26: List[float]) -> List[Dict]:
        vw = float(retrieval_config["valence_weight"])
        aw = float(retrieval_config["arousal_weight"])
        y_tau = np.asarray(tau_full26, dtype=np.float64)

        rows = []
        for row in self.rows:
            vec28 = row.get("semantic_vector_28d")
            if vec28 is None or len(vec28) < 28:
                continue

            song_va = [float(vec28[0]), float(vec28[1])]
            y_i = np.asarray(vec28[2:], dtype=np.float64)
            mu_d2 = vw * (song_va[0] - float(mu_q[0])) ** 2 + aw * (song_va[1] - float(mu_q[1])) ** 2
            tag_raw = float(np.dot(y_tau, y_i))

            rows.append(
                {
                    "song_id": row.get("song_id"),
                    "song_path": row.get("song_path"),
                    "source": row.get("source"),
                    "display_tags": row.get("display_tags", []),
                    "song_va": song_va,
                    "mu_d2": float(mu_d2),
                    "tag_raw": tag_raw,
                    "vec28": vec28,
                }
            )
        return rows

    def rank(self, mu_q: List[float], tau_full26: List[float], topk: int | None = None) -> List[Dict]:
        k = int(topk or retrieval_config["topk"])
        candidates = self._collect_candidates(mu_q, tau_full26)
        candidates.sort(key=lambda x: x["mu_d2"])

        pool_size = min(
            len(candidates),
            max(int(retrieval_config.get("va_prerank_min", 50)), k * int(retrieval_config.get("va_prerank_mult", 8))),
        )
        pool = candidates[:pool_size]

        va_rank_scores = self._rank_to_unit([x["mu_d2"] for x in pool], descending=False)
        tag_rank_scores = self._rank_to_unit([x["tag_raw"] for x in pool], descending=True)

        lambda_va = float(retrieval_config["lambda_va"])
        lambda_tag = float(retrieval_config["lambda_tag"])
        sel_tags = self._selected_tags()
        out = []
        for i, row in enumerate(pool):
            vec28 = row["vec28"]
            sel_tag_vals = {tag: float(vec28[2 + tag2idx[tag]]) for tag in sel_tags}
            top_tags = sorted(sel_tag_vals.items(), key=lambda x: x[1], reverse=True)[:5]

            score_va = float(va_rank_scores[i])
            score_tag = float(tag_rank_scores[i])
            score_total = float(lambda_va * score_va + lambda_tag * score_tag)

            out.append(
                {
                    "song_id": row.get("song_id"),
                    "song_path": row.get("song_path"),
                    "source": row.get("source"),
                    "display_tags": row.get("display_tags", []),
                    "song_va": row["song_va"],
                    "mu_d2": row["mu_d2"],
                    "score_total": score_total,
                    "score_va": score_va,
                    "score_tag": score_tag,
                    "top_tags": top_tags,
                }
            )
        out.sort(key=lambda x: (-x["score_total"], x["mu_d2"]))
        return out[:k]

    def run(self, text: str, intent: str) -> Dict:
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty")
        intent_key = self._norm_intent(intent)
        topk_val = int(retrieval_config["topk"])

        z_q = self.infer_z(text)
        mu_q, delta = self.infer_mu(z_q, intent_key)
        tau_full26, tau_selected, distances = self.infer_tau(mu_q)
        ranked = self.rank(mu_q, tau_full26, topk=topk_val)

        pool_size = min(
            sum(1 for r in self.rows if r.get("semantic_vector_28d") is not None and len(r.get("semantic_vector_28d")) >= 28),
            max(int(retrieval_config.get("va_prerank_min", 50)), topk_val * int(retrieval_config.get("va_prerank_mult", 8))),
        )

        return {
            "meta": {
                "text": text,
                "intent": intent_key,
                "q_key": retrieval_config["q_key"],
                "tag_mode": retrieval_config["tag_mode"],
                "alpha": retrieval_config["alpha"],
                "score_mode": retrieval_config.get("score_mode", "va_constrained_rerank"),
                "lambda_va": retrieval_config["lambda_va"],
                "lambda_tag": retrieval_config["lambda_tag"],
                "topk": topk_val,
                "pool_size": pool_size,
            },
            "f_1": {"text": text, "z_q": z_q},
            "f_2": {"intent": intent_key, "delta_va": delta, "mu_q": mu_q},
            "f_3": {"sel_proto": self.get_sel_proto(),},
            "f_4": {
                "q_key": retrieval_config["q_key"],
                "tag_mode": retrieval_config["tag_mode"],
                "alpha": retrieval_config["alpha"],
                "tau_selected": tau_selected,
                "distance_squared_to_mu": distances,
            },
            "f_5": {
                "topk": topk_val,
                "pool_size": pool_size,
                "lambda_va": retrieval_config["lambda_va"],
                "lambda_tag": retrieval_config["lambda_tag"],
                "results": ranked,
            },
        }

    def save_result(self, result: Dict) -> Path:
        out_path = paths["output_json"]/"demo"/"result.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

