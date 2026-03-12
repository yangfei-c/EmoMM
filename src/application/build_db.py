import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.join(src_dir, 'music_tag'))
sys.path.insert(0, os.path.join(src_dir, 'music_va'))
from music_tag.trainer import MusicTagTrainer
from music_va.trainer import MusicVATrainer

music_tag_ckpt = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\outputs\checkpoints\MusicTag\Transformer_DynamASCL\last.ckpt"
music_va_ckpt = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\outputs\checkpoints\MusicVAPred\last.ckpt"

# Jamendo 数据集路径
JAMENDO_JSON_DIR = r"D:\AAAA_audio_dataset\mtg-jamendo\splits\split-0\JSON"
JAMENDO_FEAT_DIR = r"D:\AAAA_audio_dataset\mtg-jamendo\MERT_95M_30s"
TAG_LIST_PATH = r"D:\AAAA_audio_dataset\mtg-jamendo\meta\tag_recsys.npy"
# --- 新增：Jamendo 原始音频根路径 ---
JAMENDO_AUDIO_DIR = r"D:\AAAA_audio_dataset\mtg-jamendo\audio"

# DEAM & PMEmo 数据集路径
MUSIC_VA_JSON = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\data\DEAM+PMEMO\music_va.json"
VA_FEAT_ROOT = r"D:\AAAA_audio_dataset"
# --- 新增：DEAM/PMEmo 原始音频根路径 ---
DEAM_AUDIO_DIR = r"D:\AAAA_audio_dataset\deam\mp3"
PMEMO_AUDIO_DIR = r"D:\AAAA_audio_dataset\pmemo\mp3"

# 输出路径
OUTPUT_JSON = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\data\music_db\music_db.json"

def load_mert_features(song_dir):
    if not os.path.exists(song_dir):
        return None, None

    segment_files = [f for f in os.listdir(song_dir) if f.endswith(".npy")]
    valid_segments = []

    for f in segment_files:
        feat_path = os.path.join(song_dir, f)
        try:
            feat = np.load(feat_path)
            feat = torch.from_numpy(feat).float()
            while feat.dim() > 1:
                feat = feat.mean(dim=0)
            valid_segments.append(feat)
        except Exception:
            continue

    if not valid_segments:
        return None, None

    feat_stack = torch.stack(valid_segments)

    # 1. Music Tag 特征: 768维 (直接求均值)
    feat_768 = feat_stack.mean(dim=0)

    # 2. Music VA 特征: 1536维 (均值和最大值拼接)
    feat_max = feat_stack.max(dim=0)[0]
    feat_1536 = torch.cat([feat_768, feat_max], dim=-1)

    return feat_768.unsqueeze(0), feat_1536.unsqueeze(0)


def get_top3_tags(logits, tag_list):
    """辅助函数：将模型输出转为概率并提取 Top-3 的标签，同时去除前缀"""
    preds = torch.sigmoid(logits)[0].cpu().numpy()
    top3_idx = np.argsort(preds)[::-1][:3]

    return [tag_list[idx].split('---')[-1] for idx in top3_idx]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用设备: {device}")

    print("[*] 正在加载标签字典...")
    # 标签拆分逻辑
    all_tags = np.load(TAG_LIST_PATH)
    genre_tags = all_tags[:10]
    instrument_tags = all_tags[10:17]
    mood_tags = all_tags[17:]

    print("[*] 正在加载预训练模型 (这可能需要几秒钟)...")
    tag_model = MusicTagTrainer.load_from_checkpoint(music_tag_ckpt).to(device)
    tag_model.eval()

    va_model = MusicVATrainer.load_from_checkpoint(music_va_ckpt).to(device)
    va_model.eval()

    unified_database = []

    # --- C. 处理 Jamendo (提取伪VA, 如果有真实标签则复用, 否则预测) ---
    print("\n[*] 开始处理 Jamendo 数据...")
    for split in ['train', 'validation', 'test']:
        json_file = os.path.join(JAMENDO_JSON_DIR, f"recsys-{split}.json")
        if not os.path.exists(json_file):
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            jamendo_data = json.load(f)

        for item in tqdm(jamendo_data, desc=f"Jamendo-{split}"):
            path_str = item['path'].replace('/', os.sep)
            song_dir = os.path.join(JAMENDO_FEAT_DIR, path_str)

            # --- 新增：拼接 Jamendo 真实音频路径 ---
            song_path = os.path.join(JAMENDO_AUDIO_DIR, path_str)
            if not (song_path.endswith('.mp3') or song_path.endswith('.ogg') or song_path.endswith('.wav')):
                song_path += '.mp3'  # 容错：如果没有后缀，默认补齐 .mp3

            feat_768, feat_1536 = load_mert_features(song_dir)
            if feat_1536 is None: continue

            # --- 新增：检查是否已经标注了真实 Tags ---
            raw_genres = item.get('genres', [])
            raw_instruments = item.get('instruments', [])
            raw_moods = item.get('moods/themes', item.get('moods', [])) # 兼容键名

            has_true_tags = bool(raw_genres and raw_instruments and raw_moods)

            with torch.no_grad():
                # Jamendo 必定没有 VA 标注，所以推理伪 VA
                va_logits = va_model(feat_1536.to(device))
                valence = va_logits[0, 0].item()
                arousal = va_logits[0, 1].item()

                # 如果有真实的 Tag 标注，直接使用；否则让模型推理
                if has_true_tags:
                    top3_genres = raw_genres[:3]
                    top3_instruments = raw_instruments[:3]
                    top3_moods = raw_moods[:3]
                else:
                    tag_outputs = tag_model(feat_768.to(device))
                    top3_genres = get_top3_tags(tag_outputs['genre'], genre_tags)
                    top3_instruments = get_top3_tags(tag_outputs['instrument'], instrument_tags)
                    top3_moods = get_top3_tags(tag_outputs['mood'], mood_tags)

            unified_database.append({
                "song_id": item.get('track_id', item.get('id')),
                "song_path": song_path,  # ✅ 保存可播放路径
                "source": "jamendo",
                "valence": valence,
                "arousal": arousal,
                "genres": top3_genres,
                "instruments": top3_instruments,
                "moods": top3_moods,
                "is_va_pseudo": True,
                "is_tag_pseudo": not has_true_tags # ✅ 动态判断是否为伪标签
            })

    # --- D. 处理 DEAM/PMEmo (真实VA，伪标签) ---
    print("\n[*] 开始处理 DEAM/PMEmo 数据...")
    with open(MUSIC_VA_JSON, 'r', encoding='utf-8') as f:
        va_data = json.load(f)

    for item in tqdm(va_data, desc="DEAM/PMEmo"):
        song_id = item['song_id']
        source = item['source']
        song_dir = os.path.join(VA_FEAT_ROOT, source, 'mert_30s', str(song_id))

        # --- 新增：拼接 DEAM/PMEmo 真实音频路径 ---
        if source.lower() == 'deam':
            song_path = os.path.join(DEAM_AUDIO_DIR, f"{song_id}.mp3")
        elif source.lower() == 'pmemo':
            song_path = os.path.join(PMEMO_AUDIO_DIR, f"{song_id}.mp3")
        else:
            song_path = ""

        feat_768, feat_1536 = load_mert_features(song_dir)
        if feat_768 is None: continue

        with torch.no_grad():
            # DEAM/PMEmo 必定没有 Tags 标注，所以统一推理伪 Tags
            tag_outputs = tag_model(feat_768.to(device))
            top3_genres = get_top3_tags(tag_outputs['genre'], genre_tags)
            top3_instruments = get_top3_tags(tag_outputs['instrument'], instrument_tags)
            top3_moods = get_top3_tags(tag_outputs['mood'], mood_tags)

        unified_database.append({
            "song_id": song_id,
            "song_path": song_path,  # ✅ 保存可播放路径
            "source": source,
            "valence": float(item['valence']), # 采用原数据真实 VA
            "arousal": float(item['arousal']),
            "genres": top3_genres,
            "instruments": top3_instruments,
            "moods": top3_moods,
            "is_va_pseudo": False,
            "is_tag_pseudo": True
        })

    # --- E. 保存结果 ---
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(unified_database, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 构建完成！总候选歌曲数量: {len(unified_database)}")
    print(f"📁 结果已保存至: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()