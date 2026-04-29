from __future__ import annotations

from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]

# Paths
music_db_path = project_root / "data" / "music_db" / "semantic_db.json"
text_ckpt = project_root / "outputs" / "checkpoints" / "TextVAPred_m11" / "last.ckpt"

# Text VA model
text_model_name = "xlm-roberta-base"
text_max_length = 128

# Strategy
strategies = ["maintain", "calm", "energize", "comfort"]
default_strength = 0.5

#Tag List
tags_26 = [
    "ambient", "classical", "electronic", "folk", "hiphop", "jazz",
    "metal", "pop", "rock", "soundtrack",
    "bass", "drums", "guitar", "piano", "strings", "synthesizer", "voice",
    "dark", "dream", "energetic", "epic", "film", "happy", "love", "relaxing", "sad",
]

strategy_anchors = {
    "calm": [-0.2437, -0.3755],      # ambient, relaxing, piano, dream, classical
    "energize": [0.1588, 0.2711],    # energetic, electronic, rock, pop, happy, drums
    "comfort": [-0.1940, -0.3179],   # piano, sad, classical, dream, relaxing
}

strategy_tag = {
    "maintain": [],
    "calm": ["ambient", "relaxing", "piano", "dream", "classical"],
    "energize": ["energetic", "electronic", "rock", "pop", "happy", "drums"],
    "comfort": ["piano", "sad", "classical", "dream", "relaxing"],
}

text_list = [
    {"text": "我现在有点低落，想听点能慢慢安慰我的音乐。", "intent": "comfort"},
    {"text": "I feel tired and quiet tonight.", "intent": "comfort"},
]

#diversity
diversity_weights = {
    "关闭": 0.00,
    "低": 0.15,
    "中": 0.30,
    "高": 0.50,
}
default_diversity = "低"
rerank_pool_size = 200
