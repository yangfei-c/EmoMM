import os
import sys
import json
import time
import torch
import numpy as np
import pygame
import warnings
from enum import Enum
import logging

# ==========================================
# 屏蔽不必要的警告
# ==========================================
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# 动态加载项目模块 (复用你的 inference.py 逻辑)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.join(src_dir, 'text_va'))
try:
    from text_va.trainer import TextVATrainer
    from transformers import AutoTokenizer
except ModuleNotFoundError as e:
    print(f"❌ 导入失败，请检查路径: {e}")
    sys.exit(1)


class InterventionMode(Enum):
    CATHARSIS = "catharsis"  # 宣泄/共情模式：原地停留
    HEALING = "healing"  # 疗愈/引导模式：贝塞尔曲线过渡


class EmotionTrajectoryRecommender:
    def __init__(self, db_path: str):
        print("\n[*] 正在加载音乐数据库...")
        with open(db_path, 'r', encoding='utf-8') as f:
            self.db = json.load(f)

        self.num_songs = len(self.db)
        self.va_matrix = np.zeros((self.num_songs, 2))

        for i, song in enumerate(self.db):
            self.va_matrix[i, 0] = song['valence']
            self.va_matrix[i, 1] = song['arousal']
            self.db[i]['all_tags'] = set(song['genres'] + song['instruments'] + song['moods'])

        # 数据挖掘得出的 Arousal 标签分区
        self.zone_tags = {
            "high": {"metal", "energetic", "rock", "epic", "pop", "love", "guitar"},
            "mid": {"drums", "bass", "dark", "electronic", "folk", "synthesizer", "dream", "film", "soundtrack",
                    "ambient", "hiphop", "jazz"},
            "low": {"strings", "relaxing", "voice", "sad", "piano", "classical"}
        }
        print(f"[*] 音乐库加载完毕！共 {self.num_songs} 首歌。")

    def _get_target_zone_tags(self, arousal_value: float) -> set:
        if arousal_value >= 0.58:
            return self.zone_tags["high"]
        elif 0.45 <= arousal_value < 0.58:
            return self.zone_tags["mid"]
        else:
            return self.zone_tags["low"]

    def jaccard_distance(self, tags1: set, tags2: set) -> float:
        intersection = len(tags1.intersection(tags2))
        union = len(tags1.union(tags2))
        return 1.0 - (intersection / union) if union > 0 else 1.0

    # ==========================================
    # 核心创新 1：心理学贝塞尔曲线生成
    # ==========================================
    def _generate_bezier_curve(self, p0, p1, p2, steps):
        """
        二次贝塞尔曲线: B(t) = (1-t)^2 * P0 + 2t(1-t) * P1 + t^2 * P2
        p0: 起点, p1: 控制点(决定曲线的弯曲方向), p2: 终点
        """
        t = np.linspace(0, 1, steps)
        curve = np.zeros((steps, 2))
        for i in range(steps):
            curve[i] = (1 - t[i]) ** 2 * p0 + 2 * t[i] * (1 - t[i]) * p1 + t[i] ** 2 * p2
        return curve

    # ==========================================
    # 核心创新 2：情感诊断与自动终点/控制点规划
    # ==========================================
    def auto_plan_trajectory(self, start_va: np.ndarray, steps: int):
        v, a = start_va[0], start_va[1]

        # 诊断 1: 焦虑/愤怒 (低V，高A)
        if v < 0.5 and a > 0.6:
            target_va = np.array([0.7, 0.4])  # 终点：平静放松
            # 心理学策略：先降唤醒（安抚），再升效价（开心）
            control_point = np.array([v, 0.3])
            mode_name = "降火安抚 (先降唤醒，再提效价)"

        # 诊断 2: 悲伤/抑郁 (低V，低A)
        elif v < 0.5 and a <= 0.6:
            target_va = np.array([0.7, 0.6])  # 终点：温暖轻快
            # 心理学策略：先升效价（提供情感支持），再升唤醒（鼓舞振奋）
            control_point = np.array([0.7, a])
            mode_name = "温暖鼓舞 (先给安慰，再提精神)"

        # 诊断 3: 已经很开心了 (高V)
        else:
            target_va = np.array([min(v + 0.1, 0.9), a])  # 锦上添花
            control_point = np.array([v, a])  # 接近直线
            mode_name = "锦上添花 (保持愉悦)"

        ideal_waypoints = self._generate_bezier_curve(start_va, control_point, target_va, steps)
        return ideal_waypoints, target_va, mode_name

    def recommend_trajectory(self,
                             start_va: list,
                             steps: int = 5,
                             weights: dict = {"emo": 0.5, "style": 0.3, "zone": 0.2}):

        start_va = np.array(start_va)

        # 自动推断轨迹形状
        ideal_waypoints, target_va, mode_name = self.auto_plan_trajectory(start_va, steps)

        trajectory = []
        selected_ids = set()
        current_tags = set()

        print("\n" + "=" * 65)
        print(f"🧠 [情感诊断引擎] 检测到起点 VA: [{start_va[0]:.2f}, {start_va[1]:.2f}]")
        print(f"🎯 [自动规划终点] 目标 VA: [{target_va[0]:.2f}, {target_va[1]:.2f}]")
        print(f"🛤️ [贝塞尔曲线策略] {mode_name}")
        print("=" * 65)

        for step_idx in range(steps):
            ideal_va = ideal_waypoints[step_idx]
            ideal_arousal = ideal_va[1]
            target_zone_tags = self._get_target_zone_tags(ideal_arousal)

            emotion_distances = np.linalg.norm(self.va_matrix - ideal_va, axis=1)
            candidate_indices = np.argsort(emotion_distances)[:500]

            best_song = None
            best_cost = float('inf')

            for idx in candidate_indices:
                song = self.db[idx]
                if song['song_id'] in selected_ids:
                    continue

                c_emo = emotion_distances[idx] / 1.414
                c_style = self.jaccard_distance(current_tags, song['all_tags']) if step_idx > 0 else 0.0
                match_count = len(target_zone_tags.intersection(song['all_tags']))
                c_zone = 1.0 - min(match_count / 3.0, 1.0)

                total_cost = (weights["emo"] * c_emo + weights["style"] * c_style + weights["zone"] * c_zone)

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_song = song

            trajectory.append({
                "step": step_idx + 1,
                "song_id": best_song['song_id'],
                "valence": best_song['valence'],
                "arousal": best_song['arousal'],
                "tags": list(best_song['all_tags']),
                "ideal_va": ideal_va.tolist(),
                "song_path": best_song.get('song_path', '')
            })
            selected_ids.add(best_song['song_id'])
            current_tags = best_song['all_tags']

            print(f"🟢 [Step {step_idx + 1}] 理想游标 (贝塞尔): [{ideal_va[0]:.2f}, {ideal_va[1]:.2f}]")
            print(
                f"   ↳ 选中曲目: ID {best_song['song_id']} | 真实 VA: [{best_song['valence']:>4.2f}, {best_song['arousal']:>4.2f}]")
            print(f"   ↳ 匹配标签: {list(best_song['all_tags'])}\n")

        return trajectory


def play_trajectory(path_list, play_duration=15):
    pygame.mixer.init()
    print(f"\n🎧 准备试听专属治愈轨迹，每首片段 {play_duration} 秒...")
    for item in path_list:
        song_path = item.get('song_path')
        if not song_path or not os.path.exists(song_path): continue
        print(
            f" ▶ 播放 [Step {item['step']}] - 情绪: V={item['valence']:.2f}, A={item['arousal']:.2f} | 标签: {item['tags'][:3]}")
        try:
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            time.sleep(play_duration)
            pygame.mixer.music.stop()
        except:
            pass
    pygame.mixer.quit()


# ==========================================
# 核心创新 3：加载 NLP 模型，实现真实的端到端交互
# ==========================================
def main_interactive():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TEXT_VA_CKPT = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\outputs\checkpoints\TextVAPred\last.ckpt"
    DB_PATH = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\data\music_db\music_db.json"

    print(f"[*] 正在初始化 AI 心理咨询师... (使用设备: {DEVICE})")

    # 1. 加载文本预训练模型
    try:
        text_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        text_va_model = TextVATrainer.load_from_checkpoint(TEXT_VA_CKPT, map_location=DEVICE).to(DEVICE).eval()
    except Exception as e:
        print(f"模型加载失败，请检查 checkpoint 路径: {e}")
        return

    # 2. 加载基于贝塞尔曲线的推荐器
    recommender = EmotionTrajectoryRecommender(DB_PATH)

    print("\n" + "🌟" * 25)
    print("   多模态情感音乐疗愈系统 (交互模式)   ")
    print("🌟" * 25)

    while True:
        try:
            # 尝试获取用户输入
            user_input = input("\n📝 请描述一下你此刻的心情 (输入 'q' 退出): ")
        except UnicodeDecodeError:
            # 捕获由 pygame 或 IDE 终端产生的缓冲区乱码字节 (如 0xff)
            # 并且什么都不做，直接开启下一次循环等待正常输入
            continue
        except (KeyboardInterrupt, EOFError):
            print("\n系统退出。")
            break

        if user_input.lower() in ['q', 'quit', 'exit']:
            break
        if not user_input.strip():
            continue

        # --- 文本到情感坐标的实时推理 ---
        enc = text_tokenizer(user_input, add_special_tokens=True, max_length=128,
                             padding='max_length', truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = text_va_model(enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE))

        start_v = logits[0, 0].item()
        start_a = logits[0, 1].item()
        start_va = [start_v, start_a]

        # --- 自动规划并生成推荐轨迹 ---
        trajectory = recommender.recommend_trajectory(start_va=start_va, steps=4)

        play_cmd = input("是否试听该轨迹？(y/n): ")
        if play_cmd.lower() == 'y':
            play_trajectory(trajectory, play_duration=15)


if __name__ == "__main__":
    main_interactive()