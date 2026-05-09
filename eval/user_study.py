# -*- coding: utf-8 -*-

import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from msclap import CLAP


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.music_repository import load_tracks
from core.tag_prototype import compute_tag_prototypes
from core.utils import select_nearest_tags, make_prototype_prior
from core.scoring import compute_scores, topk_indices

from text_va.inference import load_model_and_tokenizer, predict_emotion


# =========================
# Path settings
# =========================

POOL_JSON = ROOT / "eval" / "candidate_pool.json"

RESP_DIR = ROOT / "eval" / "responses"
RESP_CSV = RESP_DIR / "responses.csv"

TEXT_CKPT = ROOT / "outputs" / "checkpoints" / "TextVAPred_m11" / "last.ckpt"
TEXT_MODEL_NAME = "xlm-roberta-base"
TEXT_MAX_LENGTH = 128

CLAP_DIR = Path(r"D:\AAAA_audio_dataset\eval\clap")
CLAP_AUDIO_EMB = CLAP_DIR / "audio_embd.npz"

CLAMP3_DIR = Path(r"D:\AAAA_audio_dataset\eval\clamp3")
CLAMP3_AUDIO_EMBD_DIR = CLAMP3_DIR / "audio_embd"
CLAMP3_QUERY_CACHE = CLAMP3_DIR / "query_cache"

CLAMP3_ROOT = ROOT / "tools" / "basline" / "clamp3"
CLAMP3_CODE_DIR = CLAMP3_ROOT / "code"
EXTRACT_CLAMP3 = CLAMP3_CODE_DIR / "extract_clamp3.py"


# =========================
# Experiment settings
# =========================

TOP_K = 3
CLIP_START = 30
CLIP_DURATION = 30

TAG26 = [
    "ambient", "classical", "electronic", "folk", "hiphop", "jazz",
    "metal", "pop", "rock", "soundtrack",
    "bass", "drums", "guitar", "piano", "strings", "synthesizer", "voice",
    "dark", "dream", "energetic", "epic", "film", "happy", "love", "relaxing", "sad",
]

STRATEGY_TAGS = {
    "calm": ["ambient", "relaxing", "piano", "dream", "classical"],
    "energize": ["energetic", "electronic", "rock", "pop", "happy", "drums"],
    "comfort": ["piano", "sad", "classical", "dream", "relaxing"],
    "maintain": ["auto"],
}

STRATEGY_ANCHOR = {
    "calm": [-0.2437, -0.3755],
    "energize": [0.1588, 0.2711],
    "comfort": [-0.1940, -0.3179],
}


# =========================
# Basic utilities
# =========================

def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def clamp_va(x):
    x = float(x)
    return max(-1.0, min(1.0, x))


def parse_va_output(ret):
    """
    兼容不同 Text VA 推理函数返回格式：
    - (v, a)
    - [v, a]
    - {"valence": v, "arousal": a}
    - {"v": v, "a": a}
    - np.ndarray shape=(2,)
    """
    if isinstance(ret, dict):
        if "valence" in ret and "arousal" in ret:
            return [clamp_va(ret["valence"]), clamp_va(ret["arousal"])]
        if "v" in ret and "a" in ret:
            return [clamp_va(ret["v"]), clamp_va(ret["a"])]
        if "va" in ret:
            va = ret["va"]
            return [clamp_va(va[0]), clamp_va(va[1])]

    if isinstance(ret, (list, tuple, np.ndarray)):
        if len(ret) >= 2:
            return [clamp_va(ret[0]), clamp_va(ret[1])]

    raise ValueError(f"无法解析 Text VA 输出: {ret}")


# =========================
# Text VA
# =========================

@st.cache_resource(show_spinner="加载 Text VA 模型...")
def load_text_va_model():
    if not TEXT_CKPT.exists():
        raise FileNotFoundError(f"找不到 Text VA checkpoint: {TEXT_CKPT}")

    try:
        model, tokenizer, device = load_model_and_tokenizer(
            str(TEXT_CKPT),
            model_name=TEXT_MODEL_NAME,
        )
    except TypeError:
        model, tokenizer, device = load_model_and_tokenizer(str(TEXT_CKPT))

    return model, tokenizer, device


def predict_text_va(text):
    model, tokenizer, device = load_text_va_model()

    try:
        ret = predict_emotion(
            text,
            model,
            tokenizer,
            device,
            max_length=TEXT_MAX_LENGTH,
        )
    except TypeError:
        ret = predict_emotion(
            text,
            model,
            tokenizer,
            device,
        )

    return parse_va_output(ret)


def build_query_from_user_text(text, strategy, strength):
    z_q = np.asarray(predict_text_va(text), dtype=np.float32)

    if strategy == "maintain":
        mu_q = z_q.copy()
    else:
        anchor = np.asarray(STRATEGY_ANCHOR[strategy], dtype=np.float32)
        mu_q = (1.0 - float(strength)) * z_q + float(strength) * anchor

    return {
        "query_id": hashlib.sha1(
            f"{text}|{strategy}|{strength}".encode("utf-8")
        ).hexdigest()[:12],
        "text": text,
        "strategy": strategy,
        "strength": float(strength),
        "z_q": z_q.tolist(),
        "mu_q": mu_q.tolist(),
        "strategy_tags": STRATEGY_TAGS[strategy],
    }


# =========================
# Data loading
# =========================

@st.cache_data(show_spinner=False)
def load_pool():
    if not POOL_JSON.exists():
        raise FileNotFoundError(f"找不到候选池文件: {POOL_JSON}")

    pool = json.loads(POOL_JSON.read_text(encoding="utf-8"))
    pool_info = {str(x["song_id"]): x for x in pool}
    return pool, pool_info


@st.cache_data(show_spinner="加载 Ours 候选池...")
def load_ours_resources():
    pool, _ = load_pool()
    all_tracks = load_tracks()
    track_by_id = {str(t["song_id"]): t for t in all_tracks}

    tracks = []

    for item in pool:
        song_id = str(item["song_id"])

        if song_id not in track_by_id:
            continue

        t = dict(track_by_id[song_id])
        t["song_id"] = song_id
        t["song_path"] = item.get("song_path", t.get("song_path", ""))
        t["display_tags"] = item.get("display_tags", t.get("display_tags", []))
        t["va"] = item.get("va", t.get("va", []))

        if "tag26" not in t:
            if "semantic_vector_28d" in t and len(t["semantic_vector_28d"]) >= 28:
                t["tag26"] = t["semantic_vector_28d"][2:28]
            else:
                raise KeyError(f"track 缺少 tag26: {song_id}")

        tracks.append(t)

    if not tracks:
        raise RuntimeError("候选池为空，或 candidate_pool.json 中 song_id 无法匹配 load_tracks()。")

    z_matrix = np.asarray([t["va"] for t in tracks], dtype=np.float32)
    y_matrix = np.asarray([t["tag26"] for t in tracks], dtype=np.float32)
    prototypes = compute_tag_prototypes(tracks)

    return tracks, z_matrix, y_matrix, prototypes


@st.cache_resource(show_spinner="加载 CLAP 模型...")
def load_clap_model():
    return CLAP(version="2023", use_cuda=True)


@st.cache_data(show_spinner="加载 CLAP 音频 embedding...")
def load_clap_audio_embd():
    if not CLAP_AUDIO_EMB.exists():
        raise FileNotFoundError(f"找不到 CLAP audio embedding: {CLAP_AUDIO_EMB}")

    data = np.load(CLAP_AUDIO_EMB, allow_pickle=True)

    song_ids = [str(x) for x in data["song_ids"]]
    audio_paths = [str(x) for x in data["audio_paths"]]
    audio_emb = norm(data["audio_emb"].astype(np.float32))

    return song_ids, audio_paths, audio_emb


@st.cache_data(show_spinner="加载 CLaMP3 音频 embedding...")
def load_clamp3_audio_embd():
    if not CLAMP3_AUDIO_EMBD_DIR.exists():
        raise FileNotFoundError(f"找不到 CLaMP3 audio embedding 目录: {CLAMP3_AUDIO_EMBD_DIR}")

    song_ids = []
    vecs = []

    for p in sorted(CLAMP3_AUDIO_EMBD_DIR.rglob("*.npy")):
        x = np.load(p, allow_pickle=True).squeeze().astype(np.float32)

        if x.ndim != 1:
            continue

        song_ids.append(p.stem)
        vecs.append(x)

    if not vecs:
        raise RuntimeError(f"CLaMP3 audio embedding 为空: {CLAMP3_AUDIO_EMBD_DIR}")

    return song_ids, norm(np.stack(vecs))


# =========================
# Recommendation methods
# =========================

def recommend_ours(q, top_k):
    tracks, z_matrix, y_matrix, prototypes = load_ours_resources()

    labels = q.get("strategy_tags", [])

    if labels == ["auto"]:
        labels = select_nearest_tags(q["mu_q"], prototypes, TAG26)

    tau = make_prototype_prior(
        mu_q=q["mu_q"],
        labels=labels,
        tag26=TAG26,
        prototypes=prototypes,
        alpha=10.0,
    )

    scores = compute_scores(
        z_matrix=z_matrix,
        y_matrix=y_matrix,
        z_q=q["z_q"],
        mu_q=q["mu_q"],
        tau=tau,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )

    top_idx = topk_indices(scores["total_score"], top_k)

    rows = []

    for rank, idx in enumerate(top_idx, start=1):
        t = tracks[int(idx)]

        rows.append({
            "method": "ours",
            "rank": rank,
            "song_id": str(t["song_id"]),
            "score": round(float(scores["total_score"][idx]), 6),
            "audio_path": str(t.get("song_path", "")),
            "display_tags": t.get("display_tags", []),
            "va": t.get("va", []),
        })

    return rows


def recommend_clap(text, top_k):
    _, pool_info = load_pool()
    model = load_clap_model()
    song_ids, audio_paths, audio_emb = load_clap_audio_embd()

    text_emb = model.get_text_embeddings([text])
    text_emb = norm(to_numpy(text_emb).astype(np.float32))

    scores = audio_emb @ text_emb[0]
    top_idx = np.argsort(-scores)[:top_k]

    rows = []

    for rank, idx in enumerate(top_idx, start=1):
        song_id = song_ids[int(idx)]
        info = pool_info.get(song_id, {})

        rows.append({
            "method": "clap",
            "rank": rank,
            "song_id": song_id,
            "score": round(float(scores[int(idx)]), 6),
            "audio_path": str(info.get("song_path", audio_paths[int(idx)])),
            "display_tags": info.get("display_tags", []),
            "va": info.get("va", []),
        })

    return rows


def embed_clamp3_text(text):
    CLAMP3_QUERY_CACHE.mkdir(parents=True, exist_ok=True)

    q_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    cached_npy = CLAMP3_QUERY_CACHE / f"{q_hash}.npy"

    if cached_npy.exists():
        return np.load(cached_npy, allow_pickle=True).squeeze().astype(np.float32)

    if not EXTRACT_CLAMP3.exists():
        raise FileNotFoundError(f"找不到 CLaMP3 extract_clamp3.py: {EXTRACT_CLAMP3}")

    tmp_in = CLAMP3_QUERY_CACHE / f"tmp_in_{q_hash}"
    tmp_out = CLAMP3_QUERY_CACHE / f"tmp_out_{q_hash}"

    if tmp_in.exists():
        shutil.rmtree(tmp_in)
    if tmp_out.exists():
        shutil.rmtree(tmp_out)

    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    (tmp_in / f"{q_hash}.txt").write_text(text, encoding="utf-8")

    cmd = [
        sys.executable,
        str(EXTRACT_CLAMP3),
        str(tmp_in),
        str(tmp_out),
        "--get_global",
    ]

    subprocess.run(cmd, cwd=str(CLAMP3_CODE_DIR), check=True)

    npy_files = list(tmp_out.rglob("*.npy"))

    if not npy_files:
        raise RuntimeError("CLaMP3 没有生成 query embedding。")

    vec = np.load(npy_files[0], allow_pickle=True).squeeze().astype(np.float32)
    np.save(cached_npy, vec)

    shutil.rmtree(tmp_in)
    shutil.rmtree(tmp_out)

    return vec


def recommend_clamp3(text, top_k):
    _, pool_info = load_pool()
    song_ids, audio_emb = load_clamp3_audio_embd()

    text_vec = embed_clamp3_text(text)
    text_vec = norm(text_vec.reshape(1, -1))[0]

    if audio_emb.shape[1] != text_vec.shape[0]:
        raise ValueError(f"CLaMP3 维度不一致: audio={audio_emb.shape}, text={text_vec.shape}")

    scores = audio_emb @ text_vec
    top_idx = np.argsort(-scores)[:top_k]

    rows = []

    for rank, idx in enumerate(top_idx, start=1):
        song_id = song_ids[int(idx)]
        info = pool_info.get(song_id, {})

        rows.append({
            "method": "clamp3",
            "rank": rank,
            "song_id": song_id,
            "score": round(float(scores[int(idx)]), 6),
            "audio_path": str(info.get("song_path", "")),
            "display_tags": info.get("display_tags", []),
            "va": info.get("va", []),
        })

    return rows


def generate_all_methods(q, top_k):
    text = q["text"]

    results = {
        "ours": recommend_ours(q, top_k),
        "clap": recommend_clap(text, top_k),
        "clamp3": recommend_clamp3(text, top_k),
    }

    methods = ["ours", "clap", "clamp3"]
    random.shuffle(methods)

    label_to_method = {
        "A": methods[0],
        "B": methods[1],
        "C": methods[2],
    }

    return results, label_to_method


# =========================
# Response saving
# =========================

def append_response(rows):
    RESP_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "participant_id",
        "query_id",
        "query_text",
        "strategy",
        "strength",
        "method_label",
        "true_method",
        "regulation_fit",
        "helpfulness",
        "satisfaction",
        "preferred_label",
        "preferred_method",
        "top_song_ids",
        "top_audio_paths",
        "top_display_tags",
        "top_va",
        "top_scores",
        "z_q",
        "mu_q",
    ]

    exists = RESP_CSV.exists()

    with RESP_CSV.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerows(rows)


# =========================
# UI
# =========================

def render_audio_item(item):
    audio_path = Path(str(item["audio_path"]))

    st.caption(f"音频 {item['rank']}")

    if not audio_path.exists():
        st.warning("该音频文件未找到，无法播放。")
        return

    try:
        st.audio(
            str(audio_path),
            start_time=CLIP_START,
            end_time=CLIP_START + CLIP_DURATION,
        )
    except TypeError:
        st.audio(
            str(audio_path),
            start_time=CLIP_START,
        )


def main():
    st.set_page_config(
        page_title="音乐情绪调节推荐用户研究",
        layout="wide",
    )

    st.title("音乐情绪调节推荐用户研究")
    st.caption("请根据自己的真实听感评价三组匿名推荐，不要猜测推荐系统的意图。")

    with st.sidebar:
        st.header("实验设置")

        participant_id = st.text_input("被试编号", value="P001")

        strategy = st.selectbox(
            "调节目标",
            options=["calm", "energize", "comfort", "maintain"],
            format_func=lambda x: {
                "calm": "平静 / 安抚",
                "energize": "振作 / 激活",
                "comfort": "安慰 / 陪伴",
                "maintain": "保持当前情绪",
            }[x],
        )

        strength = st.select_slider(
            "调节强度",
            options=[0.3, 0.5, 0.7],
            value=0.5,
        )

        generate_btn = st.button("生成三种推荐", type="primary")

    text = st.text_area(
        "请描述你当前的情绪状态，以及希望音乐帮你达到的情绪目标：",
        value="我今天有些疲惫和低落，希望音乐能让我慢慢放松下来。",
        height=120,
    )

    if generate_btn:
        if not participant_id.strip():
            st.error("请输入被试编号。")
            return

        if not text.strip():
            st.error("请输入情绪描述。")
            return

        with st.spinner("后端正在实时生成三种推荐..."):
            q = build_query_from_user_text(
                text=text.strip(),
                strategy=strategy,
                strength=strength,
            )

            results, label_to_method = generate_all_methods(q, TOP_K)

        st.session_state["q"] = q
        st.session_state["results"] = results
        st.session_state["label_to_method"] = label_to_method
        st.session_state["participant_id"] = participant_id.strip()

    if "results" not in st.session_state:
        st.info("请输入情绪描述，然后点击左侧“生成三种推荐”。")
        return

    q = st.session_state["q"]
    results = st.session_state["results"]
    label_to_method = st.session_state["label_to_method"]
    participant_id = st.session_state["participant_id"]

    st.subheader("匿名推荐结果")

    cols = st.columns(3)

    for col, label in zip(cols, ["A", "B", "C"]):
        method = label_to_method[label]

        with col:
            st.markdown(f"### 方法 {label}")

            for item in results[method]:
                render_audio_item(item)

    st.divider()
    st.subheader("评分")

    with st.form("rating_form"):
        ratings = {}

        for label in ["A", "B", "C"]:
            st.markdown(f"#### 方法 {label}")

            fit = st.slider(
                f"Regulation Fit：这组音乐符合当前情绪场景和调节目标。({label})",
                min_value=1,
                max_value=7,
                value=4,
                key=f"fit_{label}",
            )

            helpfulness = st.slider(
                f"Helpfulness：这组音乐有助于我达到描述中的情绪目标。({label})",
                min_value=1,
                max_value=7,
                value=4,
                key=f"help_{label}",
            )

            satisfaction = st.slider(
                f"Satisfaction：我对这组音乐推荐整体满意。({label})",
                min_value=1,
                max_value=7,
                value=4,
                key=f"sat_{label}",
            )

            ratings[label] = {
                "regulation_fit": fit,
                "helpfulness": helpfulness,
                "satisfaction": satisfaction,
            }

        preferred_label = st.radio(
            "在 A/B/C 三组推荐中，你最愿意选择哪一组？",
            ["A", "B", "C"],
            horizontal=True,
        )

        submitted = st.form_submit_button("保存本轮评分")

    if submitted:
        preferred_method = label_to_method[preferred_label]
        rows = []

        for label in ["A", "B", "C"]:
            method = label_to_method[label]
            recs = results[method]

            rows.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "participant_id": participant_id,
                "query_id": q["query_id"],
                "query_text": q["text"],
                "strategy": q["strategy"],
                "strength": q["strength"],
                "method_label": label,
                "true_method": method,
                "regulation_fit": ratings[label]["regulation_fit"],
                "helpfulness": ratings[label]["helpfulness"],
                "satisfaction": ratings[label]["satisfaction"],
                "preferred_label": preferred_label,
                "preferred_method": preferred_method,
                "top_song_ids": json.dumps([x["song_id"] for x in recs], ensure_ascii=False),
                "top_audio_paths": json.dumps([x["audio_path"] for x in recs], ensure_ascii=False),
                "top_display_tags": json.dumps([x["display_tags"] for x in recs], ensure_ascii=False),
                "top_va": json.dumps([x["va"] for x in recs], ensure_ascii=False),
                "top_scores": json.dumps([x["score"] for x in recs], ensure_ascii=False),
                "z_q": json.dumps(q["z_q"], ensure_ascii=False),
                "mu_q": json.dumps(q["mu_q"], ensure_ascii=False),
            })

        append_response(rows)
        st.success("评分已保存。")


if __name__ == "__main__":
    main()