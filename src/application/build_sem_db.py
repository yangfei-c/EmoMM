import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

# 动态加载模块
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.join(src_dir, 'music_tag'))
sys.path.insert(0, os.path.join(src_dir, 'music_va'))

from music_tag.trainer import MusicTagTrainer
from music_va.trainer import MusicVATrainer

# ==========================================
# 路径配置
# ==========================================
m_tag_ckpt = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\outputs\checkpoints\MusicTag\Transformer_DynamASCL\last.ckpt"
m_va_ckpt = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\outputs\checkpoints\MusicVAPred\last.ckpt"

J_JSON_DIR = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\data\MTG-Jamendo"
J_FEAT_DIR = r"D:\AAAA_audio_dataset\mtg-jamendo\MERT_95M_30s"
TAG_LIST_PATH = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\data\MTG-Jamendo\tags_26.npy"
J_AUDIO_DIR = r"D:\AAAA_audio_dataset\mtg-jamendo\audio"

# 最终输出的数据库文件路径
OUTPUT_JSON = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\data\music_db\semantic_db.json"


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

    # Music Tag 特征: 768维 (均值)
    feat_768 = feat_stack.mean(dim=0)
    # Music VA 特征: 1536维 (均值+最大值拼接)
    feat_max = feat_stack.max(dim=0)[0]
    feat_1536 = torch.cat([feat_768, feat_max], dim=-1)

    return feat_768.unsqueeze(0), feat_1536.unsqueeze(0)


def tags_for_display(logits, tag_list):
    """ Top-3标签"""
    preds = torch.sigmoid(logits)[0].cpu().numpy()
    top3_idx = np.argsort(preds)[::-1][:3]
    return [tag_list[idx].split('---')[-1] for idx in top3_idx]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用设备: {device}")

    print("[*] 正在加载标签字典...")
    all_tags = np.load(TAG_LIST_PATH)
    genre_tags = all_tags[:10]
    instrument_tags = all_tags[10:17]
    mood_tags = all_tags[17:]

    print("[*] 正在加载预训练模型...")
    tag_model = MusicTagTrainer.load_from_checkpoint(m_tag_ckpt, map_location=device).to(device)
    tag_model.eval()

    va_model = MusicVATrainer.load_from_checkpoint(m_va_ckpt, map_location=device).to(device)
    va_model.eval()

    save_db = []

    # 直接读取合并好的单一 jamendo.json
    jamendo_json = os.path.join(J_JSON_DIR, "jamendo.json")
    if not os.path.exists(jamendo_json):
        print(f"❌ 找不到文件: {jamendo_json}！")
        return

    print("\n[*] 开始处理 Jamendo 数据 (构建 28 维语义向量)...")
    with open(jamendo_json, 'r', encoding='utf-8') as f:
        jamendo_data = json.load(f)

    for item in tqdm(jamendo_data, desc="MTG-Jamendo"):
        path_str = item['path'].replace('/', os.sep)
        song_dir = os.path.join(J_FEAT_DIR, path_str)

        song_path = os.path.join(J_AUDIO_DIR, path_str)
        if not (song_path.endswith('.mp3') or song_path.endswith('.ogg') or song_path.endswith('.wav')):
            song_path += '.mp3'

        feat_768, feat_1536 = load_mert_features(song_dir)

        if feat_1536 is None or feat_768 is None:
            continue

        with torch.no_grad():
            # va
            va_logits = va_model(feat_1536.to(device))
            valence = va_logits[0, 0].item()
            arousal = va_logits[0, 1].item()

            # tags
            tag_outputs = tag_model(feat_768.to(device))
            genre_probs = torch.sigmoid(tag_outputs['genre'])[0].cpu().numpy()  # 10维
            inst_probs = torch.sigmoid(tag_outputs['instrument'])[0].cpu().numpy()  # 7维
            mood_probs = torch.sigmoid(tag_outputs['mood'])[0].cpu().numpy()  # 9维

            # --- 获取标签 ---
            display_genres = tags_for_display(tag_outputs['genre'], genre_tags)
            display_insts = tags_for_display(tag_outputs['instrument'], instrument_tags)
            display_moods = tags_for_display(tag_outputs['mood'], mood_tags)

        # --- 拼接28维向量 ---
        vector_28d: np.ndarray = np.concatenate(([valence, arousal], genre_probs, inst_probs, mood_probs))

        save_db.append({
            "song_id": item.get('track_id', item.get('id')),
            "source": "jamendo",
            "semantic_vector_28d": vector_28d.tolist(),
            "display_tags": display_genres + display_insts + display_moods,
            "song_path": song_path,
        })

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(save_db, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 语义数据库构建完成！总候选歌曲数量: {len(save_db)}")
    print(f"📁 结果已保存至: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()