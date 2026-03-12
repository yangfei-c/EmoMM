import json
import numpy as np
import time
import os
import pygame


class EmotionTrajectoryRecommender:
    def __init__(self, db_path: str):
        print("[*] 正在加载音乐数据库...")
        start_time = time.time()
        with open(db_path, 'r', encoding='utf-8') as f:
            self.db = json.load(f)
        # 构建VA矩阵
        self.num_songs = len(self.db)
        self.va_matrix: np.ndarray = np.zeros((self.num_songs, 2))

        for i, song in enumerate(self.db):
            self.va_matrix[i, 0] = song['valence']
            self.va_matrix[i, 1] = song['arousal']
            song['all_tags'] = set(song['genres'] + song['instruments'] + song['moods'])
        print(f"[*] 加载完毕！共 {self.num_songs} 首候选歌曲。耗时: {time.time() - start_time:.2f} 秒")

    def jaccard_distance(self, tags1: set, tags2: set) -> float:
        """计算两个标签集合的 Jaccard 距离 (值越小，风格越相似)"""
        if not tags1 and not tags2:
            return 0.0
        intersection = len(tags1.intersection(tags2))
        union = len(tags1.union(tags2))
        return 1.0 - (intersection / union) if union > 0 else 1.0

    def recommend_trajectory(self,
                             start_va: list,
                             target_va: list,
                             steps: int = 5,
                             alpha: float = 0.7,
                             beta: float = 0.3,
                             user_preferred_tags: list = None):
        """
        核心算法：启发式情感轨迹寻路 (增加上帝视角打印)
        """
        start_va = np.array(start_va)
        target_va = np.array(target_va)
        user_tags = set(user_preferred_tags) if user_preferred_tags else set()

        # 1. 生成理想情绪航点 (线性插值)
        ideal_waypoints: np.ndarray = np.linspace(start_va, target_va, steps)

        trajectory = []
        selected_ids = set()
        current_tags = user_tags

        print("\n" + "=" * 50)
        print("🚀 [上帝视角] 启发式寻路算法内部过程追踪")
        print("=" * 50)

        for step_idx in range(steps):
            ideal_va = ideal_waypoints[step_idx]
            print(f"\n🟢 【Step {step_idx + 1}】")
            print(f"   📍 算法生成理想航点 (Ideal_VA): [{ideal_va[0]:.3f}, {ideal_va[1]:.3f}]")
            print(f"   🔗 当前用于平滑匹配的基准标签 (Current Tags): {list(current_tags)}")

            emotion_distances = np.linalg.norm(self.va_matrix - ideal_va, axis=1)
            candidate_indices = np.argsort(emotion_distances)[:500]
            print(f"   🔍 空间粗筛: 已从 {self.num_songs} 首歌中找出距离 Ideal_VA 最近的 Top-500")

            best_song = None
            best_cost = float('inf')
            best_emo_cost = 0.0
            best_style_cost = 0.0

            for idx in candidate_indices:
                song = self.db[idx]
                if song['song_id'] in selected_ids:
                    continue

                emo_cost = emotion_distances[idx] / 1.414
                smoothness_cost = self.jaccard_distance(current_tags, song['all_tags'])
                if user_tags:
                    preference_cost = 0.0 if len(user_tags.intersection(song['all_tags'])) > 0 else 1.0
                else:
                    preference_cost = 0.0

                style_cost = 0.5 * smoothness_cost + 0.5 * preference_cost
                total_cost = alpha * emo_cost + beta * style_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_song = song
                    best_emo_cost = emo_cost
                    best_style_cost = style_cost

            # 记录这一步的结果
            trajectory.append({
                "step": step_idx + 1,
                "song_id": best_song['song_id'],
                "valence": best_song['valence'],
                "arousal": best_song['arousal'],
                "tags": list(best_song['all_tags']),
                "ideal_va": ideal_va.tolist(),
                "cost": float(best_cost),
                "song_path": best_song.get('song_path')
            })
            selected_ids.add(best_song['song_id'])
            current_tags = best_song['all_tags']  # 核心：基准变更为当前选中的歌

            print(f"   🏆 经过双重成本计算，选中天命之曲 -> ID: {best_song['song_id']}")
            print(f"      * 它的真实 VA : [{best_song['valence']:.3f}, {best_song['arousal']:.3f}]")
            print(f"      * 它的标签属性: {list(best_song['all_tags'])}")
            print(
                f"      * 💰 成本结算  : 情感代价(E)={best_emo_cost:.3f}, 风格代价(S)={best_style_cost:.3f} --> 最终总代价(Cost)={best_cost:.3f}")
            print("-" * 50)

        return trajectory


def play_trajectory(path_list, play_duration=15):
    pygame.mixer.init()
    print(f"\n🎧 准备开始试听轨迹，每首试听 {play_duration} 秒...")
    for item in path_list:
        song_path = item.get('song_path')
        step = item['step']
        if not song_path or not os.path.exists(song_path):
            print(f" [Step {step}] ⚠️ 音频文件不存在，跳过试听: {song_path}")
            continue
        print(
            f" ▶ 正在播放 [Step {step}] - ID: {item['song_id']} | 情绪: V={item['valence']:.2f}, A={item['arousal']:.2f}")
        try:
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            time.sleep(play_duration)
            pygame.mixer.music.stop()
        except Exception as e:
            print(f" [Step {step}] ❌ 播放出错: {e}")
    pygame.mixer.quit()
    print("⏹ 轨迹试听结束。\n")


if __name__ == "__main__":
    DB_PATH = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\data\music_db\music_db.json"  # 请确保路径正确
    recommender = EmotionTrajectoryRecommender(DB_PATH)

    # 我们精选了 3 个从你 26 个标签里选出的 Tag 用于缓解焦虑
    scenarios = [
        {
            "name": "场景 1: 缓解重度焦虑 (实例追踪)",
            "desc": "高唤醒/负效价 -> 平静放松",
            "start_va": [0.2, 0.9],
            "target_va": [0.7, 0.2],
            "tags": ["ambient", "piano", "relaxing"]
        }
    ]

    for idx, sc in enumerate(scenarios):
        print(f"\n{'=' * 60}")
        print(f"▶ {sc['name']}")
        print(f"  起点 VA: {sc['start_va']} | 终点 VA: {sc['target_va']}")
        print(f"  偏好风格: {sc['tags']}")
        print(f"{'-' * 60}")

        path = recommender.recommend_trajectory(
            start_va=sc['start_va'],
            target_va=sc['target_va'],
            steps=5,
            alpha=0.6,
            beta=0.4,
            user_preferred_tags=sc['tags']
        )

        # 考虑到前面打印已经很详细，这里就不再重复打印总结了
        input("\n按回车键 [Enter] 开始播放这条情感轨迹...")
        play_trajectory(path, play_duration=15)