import os
import sys
import json
import warnings
import logging
# ==========================================
# 屏蔽不必要的警告和日志输出
# ==========================================
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
import torch
import numpy as np
import random as std_random

#将包导入到系统中
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '../music_tag'))
sys.path.insert(0, os.path.join(current_dir, '../music_va'))
sys.path.insert(0, os.path.join(current_dir, '../text_va'))
try:
    from music_tag.trainer import MusicTagTrainer
    from music_va.trainer import MusicVATrainer
    from text_va.trainer import TextVATrainer
    from transformers import AutoTokenizer
except ModuleNotFoundError as e:
    print(f"❌ 导入失败，请检查路径: {e}")
    sys.exit(1)
test_samples=1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MTG-Jamendo ---
JAMENDO_JSON = r"D:\AAAA_audio_dataset\mtg-jamendo\splits\split-0\JSON\recsys-test.json"
JAMENDO_FEAT_DIR = r"D:\AAAA_audio_dataset\mtg-jamendo\MERT_95M_30s"
TAG_LIST_PATH = r"D:\AAAA_audio_dataset\mtg-jamendo\meta\tag_recsys.npy"

#DEAM/PMEmo 路径配置
MUSIC_VA_JSON = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\data\music_va.json"
VA_FEAT_ROOT = r"D:\AAAA_audio_dataset"

#预训练权重
MUSIC_TAG_CKPT = r"..\outputs\checkpoints\MusicTag\last.ckpt"
MUSIC_VA_CKPT = r"..\outputs\checkpoints\MusicVAPred\last.ckpt"
TEXT_VA_CKPT = r"..\outputs\checkpoints\TextVAPred\last.ckpt"

print("\n" + "=" * 65)
print("🚀 正在初始化系统...".center(55))
print("=" * 65)
print(f"[*] 计算设备 : {DEVICE}")

print("[*] 正在加载 Music Tag 模型...")
tag_model = MusicTagTrainer.load_from_checkpoint(MUSIC_TAG_CKPT, map_location=DEVICE).to(DEVICE).eval()
print("[*] 正在加载 Music VA 模型...")
music_va_model = MusicVATrainer.load_from_checkpoint(MUSIC_VA_CKPT, map_location=DEVICE).to(DEVICE).eval()
print("[*] 正在加载 Text VA 模型 (XLM-RoBERTa)...")
text_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
text_va_model = TextVATrainer.load_from_checkpoint(TEXT_VA_CKPT, map_location=DEVICE).to(DEVICE).eval()

# --- 加载标签字典 ---
all_tags = np.load(TAG_LIST_PATH)
genre_tags = all_tags[:10]
instrument_tags = all_tags[10:17]
mood_tags = all_tags[17:]


def get_top3_tags(logits, tag_list):
    preds = torch.sigmoid(logits)[0].cpu().numpy()
    top3_idx = np.argsort(preds)[::-1][:3]
    return [tag_list[idx].split('---')[-1] for idx in top3_idx]


def load_audio_features(song_dir):
    segment_files = [f for f in os.listdir(song_dir) if f.endswith(".npy")]
    valid_segments = []
    for f in segment_files:
        try:
            feat = np.load(os.path.join(song_dir, f))
            feat = torch.from_numpy(feat).float()
            while feat.dim() > 1: feat = feat.mean(dim=0)
            valid_segments.append(feat)
        except Exception:
            continue
    feat_stack = torch.stack(valid_segments)
    feat_768 = feat_stack.mean(dim=0)
    feat_1536 = torch.cat([feat_768, feat_stack.max(dim=0)[0]], dim=-1)
    return feat_768.unsqueeze(0), feat_1536.unsqueeze(0)

# ---------------- Music Tag 推理 (Jamendo) ----------------
print("[*] 正在抽取 Jamendo 音乐进行 Tag 推理...")
with open(JAMENDO_JSON, 'r', encoding='utf-8') as f:
    jamendo_data = json.load(f)
valid_jamendo = [s for s in jamendo_data if s.get('genres') and s.get('instruments') and s.get('moods/themes')]
test_tag_songs = std_random.sample(valid_jamendo, test_samples)

tag_results = []
for idx, song in enumerate(test_tag_songs):
    path_str = song['path'].replace('/', os.sep)
    song_dir = os.path.join(JAMENDO_FEAT_DIR, path_str)
    feat_768, _ = load_audio_features(song_dir)

    if feat_768 is None: continue
    with torch.no_grad():
        tag_outputs = tag_model(feat_768.to(DEVICE))
        pred_g = get_top3_tags(tag_outputs['genre'], genre_tags)
        pred_i = get_top3_tags(tag_outputs['instrument'], instrument_tags)
        pred_m = get_top3_tags(tag_outputs['mood'], mood_tags)

    tag_results.append({
        'id': song.get('track_id', f"Jamendo-{idx}"),
        'true_tags': f"{song['genres'][0]}, {song['instruments'][0]}, {song['moods/themes'][0]}",
        'pred_tags': f"{pred_g[0]}, {pred_i[0]}, {pred_m[0]}"
    })

# ----------------  Music VA 推理 (DEAM/PMEmo) ----------------
print("[*] 正在抽取 DEAM/PMEmo 音乐进行 VA 推理...")
with open(MUSIC_VA_JSON, 'r', encoding='utf-8') as f:
    va_data = json.load(f)
test_va_songs = std_random.sample(va_data, test_samples)

va_results = []
for idx, song in enumerate(test_va_songs):
    song_id = song['song_id']
    source = song['source']
    true_v = float(song['valence'])
    true_a = float(song['arousal'])

    song_dir = os.path.join(VA_FEAT_ROOT, source, 'mert_30s', str(song_id))
    _, feat_1536 = load_audio_features(song_dir)

    if feat_1536 is None: continue
    with torch.no_grad():
        va_logits = music_va_model(feat_1536.to(DEVICE))
        pred_v, pred_a = va_logits[0, 0].item(), va_logits[0, 1].item()

    va_results.append({
        'id': f"{source.upper()}-{song_id}",
        'true_v': true_v, 'true_a': true_a,
        'pred_v': pred_v, 'pred_a': pred_a
    })

# ---------------- Text VA 推理 ----------------
print("[*] 正在执行中英文本语言推理...")
zh_text = "今天天气真好，和朋友一起去郊游，心情非常舒畅！"
en_text = "The weather is so nice today. Went on a picnic with friends, feel great!"
zh_text1 = "考试没过，我好难过"
en_text1 = "I failed the exam, I'm so sad."


def predict_text(text):
    enc = text_tokenizer(text, add_special_tokens=True, max_length=128, padding='max_length', truncation=True,
                         return_tensors='pt')
    with torch.no_grad():
        logits = text_va_model(enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE))
    return logits[0, 0].item(), logits[0, 1].item()

zh_v, zh_a = predict_text(zh_text)
en_v, en_a = predict_text(en_text)
zh_v1, zh_a1 = predict_text(zh_text1)
en_v1, en_a1 = predict_text(en_text1)

#清屏

print(" 【音乐多标签分类 (Jamendo)】")
print("-" * 70)
for idx, res in enumerate(tag_results):
    print(f"  音乐测试样本 {idx + 1}  [ID: {res['id']}]")
    print(f"   ├─ 真实标签   : [ {res['true_tags']} ]")
    print(f"   └─ 模型预测   : [ {res['pred_tags']} ]\n")

print(" 【音乐情感回归 (DEAM & PMEmo)】")
print("-" * 70)
for idx, res in enumerate(va_results):
    # 计算偏差
    diff_v = abs(res['true_v'] - res['pred_v'])
    diff_a = abs(res['true_a'] - res['pred_a'])

    print(f"  情感测试样本 {idx + 1}  [ID: {res['id']}]")
    print(f"   ├─ 真实情感  : Valence = {res['true_v']:.4f} | Arousal = {res['true_a']:.4f}")
    print(f"   ├─ 模型预测  : Valence = {res['pred_v']:.4f} | Arousal = {res['pred_a']:.4f}")
    print(f"   └─ 绝对误差  : V_diff = {diff_v:.4f}  | A_diff = {diff_a:.4f}\n")

print(" 【文本跨语言情感对齐推理】")
print("-" * 70)

print(f" 🇨🇳 中文输入 : \"{zh_text}\"")
print(f"   └─ 预测情感 (VA)   : 效价 Valence = {zh_v:.4f} | 唤醒度 Arousal = {zh_a:.4f}\n")
print(f" 🇺🇸 英文输入 : \"{en_text}\"")
print(f"   └─ 预测情感 (VA)   : 效价 Valence = {en_v:.4f} | 唤醒度 Arousal = {en_a:.4f}\n")

print(f" 🇨🇳 中文输入 : \"{zh_text1}\"")
print(f"   └─ 预测情感 (VA)   : 效价 Valence = {zh_v1:.4f} | 唤醒度 Arousal = {zh_a1:.4f}\n")
print(f" 🇺🇸 英文输入 : \"{en_text1}\"")
print(f"   └─ 预测情感 (VA)   : 效价 Valence = {en_v1:.4f} | 唤醒度 Arousal = {en_a1:.4f}\n")

print("=" * 70)
print("✅ 所有模块推理完毕！".center(65))
print("=" * 70 + "\n")