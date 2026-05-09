from __future__ import annotations

from typing import Dict, List

from core.config import (
    strength,
    strategies,
    strategy_anchors,
    strategy_tag,
    tags_26,
    text_ckpt,
    text_max_length,
    text_model_name,
)
from core.utils import clip_va, interpolate_va, make_tag_prior
from text_va.inference import load_model_and_tokenizer, predict_emotion


class UserStateEstimator:
    def __init__(self) -> None:
        self._model_pack = None

    def _load_model(self):
        if self._model_pack is not None:
            return self._model_pack
        if not text_ckpt.exists():
            raise FileNotFoundError(f"Text VA checkpoint not found: {text_ckpt}")
        self._model_pack = load_model_and_tokenizer(str(text_ckpt), model_name=text_model_name)
        return self._model_pack

    def infer_z_q(self, text: str) -> List[float]:
        model, tokenizer, device = self._load_model()
        v, a = predict_emotion(text, model, tokenizer, device, max_length=text_max_length)
        return clip_va([v, a])

    def infer_mu_q(self, z_q, strategy: str, strength: float = strength) -> Dict[str, object]:
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        z_q = clip_va(z_q)
        s = max(0.0, min(1.0, float(strength)))

        z_anchor = z_q if strategy == "maintain" else strategy_anchors[strategy]
        mu_q = interpolate_va(z_q, z_anchor, s)
        tau = make_tag_prior(strategy_tag.get(strategy, []), tags_26)

        return {
            "z_q": z_q,
            "strategy": strategy,
            "strength": s,
            "anchor": [float(z_anchor[0]), float(z_anchor[1])],
            "mu_q": mu_q,
            "delta": [float(mu_q[0] - z_q[0]), float(mu_q[1] - z_q[1])],
            "tag_prior": tau,
        }

    def process(self, text: str, strategy: str, strength: float = strength) -> Dict[str, object]:
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty")
        z_q = self.infer_z_q(text)
        return self.infer_mu_q(z_q, strategy=strategy, strength=strength)


if __name__ == "__main__":
    from core.config import text_list

    estimator = UserStateEstimator()
    for item in text_list:
        result = estimator.process(item["text"], strategy=item["intent"], strength=strength)
        print(f"Text: {item['text']}")
        print(f"Strategy: {item['intent']}")
        print(f"current_va: {result['z_q']}")
        print(f"anchor: {result['anchor']}")
        print(f"target_q: {result['mu_q']}\n")
