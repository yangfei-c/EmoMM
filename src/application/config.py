from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
APP_DIR = SRC / "application"

paths = {
    "text_ckpt": ROOT / "outputs" / "checkpoints" / "TextVAPred_m11" / "last.ckpt",
    "semantic_db": ROOT / "data" / "music_db" / "semantic_db.json",
    "prototype_json": APP_DIR / "outputs" / "tag_va_prototypes_topq.json",
    "separation_json": APP_DIR / "outputs" / "analysis" / "tag_separation_q095.json",
    "output_json": APP_DIR / "outputs" / "analysis" / "demo_result.json",
    "obs_root": ROOT / "src" / "application" / "outputs" / "analysis",
}
PATHS = paths

text_config = {
    "model_name": "xlm-roberta-base",
    "max_length": 128,
}
TEXT_CFG = text_config

tag_choices = {
    # Legacy subset: kept for backward compatibility, not used in current main experiments.
    "filtered8": [
        "dark", "dream", "energetic", "epic", "happy", "love", "relaxing", "sad",
    ],
    "core10": [
        "dark", "dream", "energetic", "epic", "happy", "love", "relaxing", "sad",
        "ambient", "piano",
    ],
}
TAG_SUBSETS = tag_choices

retrieval_config = {
    "q_key": "0.95",
    "tag_mode": "core10",  # active experiment modes: full26 | core10
    "alpha": 5.0,  # Formula 4: exp(-alpha * ||mu-c||^2)

    # Formula 5: VA-constrained rerank (stage1 by VA distance, stage2 by weighted rank fusion)
    "score_mode": "va_constrained_rerank",
    "lambda_va": 0.80,
    "lambda_tag": 0.20,
    "valence_weight": 1.0,
    "arousal_weight": 1.0,
    "topk": 10,
    "va_prerank_mult": 8,
    "va_prerank_min": 50,
}
RET_CFG = retrieval_config


intent_strategy = {
    "validate": {"type": "zero"},
    "soothe": {"type": "delta", "dv_pos": 0.18, "da_neg": 0.35},
    "uplift": {"type": "target", "target_v": 0.45, "target_a": 0.10, "kv": 0.45, "ka": 0.35},
    "energize": {"type": "delta", "dv_pos": 0.22, "da_pos": 0.28},
}
INTENT_GAIN = intent_strategy

tags_26 = [
    "ambient", "classical", "electronic", "folk", "hiphop", "jazz", "metal", "pop", "rock", "soundtrack",
    "bass", "drums", "guitar", "piano", "strings", "synthesizer", "voice",
    "dark", "dream", "energetic", "epic", "film", "happy", "love", "relaxing", "sad",
]
TAGS_26 = tags_26

tag2idx = {t: i for i, t in enumerate(tags_26)}
TAG_TO_IDX = tag2idx

