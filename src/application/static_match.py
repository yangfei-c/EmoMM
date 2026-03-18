import json
import numpy as np
import time

from psycho_mapping import (
    determine_intervention_strategy,
    calculate_penalty,
    INST_IDX,
    INSTRUMENT_EFFECTS
)


class PsychologyGuidedRec:
    def __init__(self, db_path: str):
        print("[*] 正在加载 28维语义音乐数据库与心理学干预引擎...")
        start_time = time.time()
        with open(db_path, 'r', encoding='utf-8') as f:
            self.db = json.load(f)
        self.song_matrix = np.array([song['semantic_vector_28d'] for song in self.db])
        self.va_matrix = self.song_matrix[:, 0:2]
        print(f"[*] 加载完毕！共 {len(self.db)} 首歌。耗时: {time.time() - start_time:.2f} 秒")

    def _retrieve_song(self, target_va, personality, pref_instruments, exclude_ids):
        va_distances = np.linalg.norm(self.va_matrix - target_va, axis=1)
        sigma = 0.5
        score_emotion = np.exp(- (va_distances ** 2) / (2 * sigma ** 2))

        best_score = -float('inf')
        best_song = None

        for idx, song in enumerate(self.db):
            if song['song_id'] in exclude_ids:
                continue

            vec_28d = song['semantic_vector_28d']
            score_personality = calculate_penalty(personality, vec_28d)
            score_instrument = 0.0
            for inst in pref_instruments:
                if inst in INST_IDX:
                    inst_prob = vec_28d[INST_IDX[inst]]
                    score_instrument += inst_prob * 3.0
                    if inst in INSTRUMENT_EFFECTS:
                        score_emotion[idx] += inst_prob * 0.1
            total_score = (score_emotion[idx] * 5.0) + score_personality + score_instrument
            if total_score > best_score:
                best_score = total_score
                best_song = song

        return best_song

    def generate_playlist(self, v_user, a_user, personality_scores):
        """
        生成基于 ISO 原则的三阶段动态干预歌单 (Trajectory Playlist)
        """
        print(f"\n用户当前状态: Valence={v_user:.2f}, Arousal={a_user:.2f}")
        print(f"用户人格画像: {personality_scores}")

        strategy_info = determine_intervention_strategy(v_user, a_user, personality_scores)
        target_va_final = strategy_info['target_va']
        print(f"[*] 匹配策略: {strategy_info['strategy']}")
        print(f"[*] 最终心境目标: Valence={target_va_final[0]:.2f}, Arousal={target_va_final[1]:.2f}")
        playlist = []
        exclude_ids = set()  # 防止推荐重复的歌曲

        target_1 = np.array([v_user, a_user])  # 第一首歌完全贴合用户当前情绪
        # 负面情绪时，优先使用钢琴/弦乐等具有共情和抚慰作用的乐器
        pref_inst_1 = ["piano", "strings"] if v_user < 0.4 else []

        song_1 = self._retrieve_song(target_1, personality_scores, pref_inst_1, exclude_ids)
        if song_1:
            exclude_ids.add(song_1['song_id'])
            playlist.append({
                "stage": "第一阶段：共情与接纳",
                "track": song_1,
                "reason": "同质原则起点：这首歌的声学特征贴合你当下的心境，给你一个安全的空间去释放和整理思绪。"
            })
        target_2 = (np.array([v_user, a_user]) + target_va_final) / 2.0
        pref_inst_2 = ["guitar"]
        song_2 = self._retrieve_song(target_2, personality_scores, pref_inst_2, exclude_ids)
        if song_2:
            exclude_ids.add(song_2['song_id'])
            playlist.append({
                "stage": "第二阶段：过渡与重构",
                "track": song_2,
                "reason": "情感过渡：节奏与音色逐渐发生变化，引导大脑开始清理负面思绪，走向平静。"
            })
        target_3 = target_va_final
        pref_inst_3 = strategy_info['pref_instruments']  # 采用策略引擎推荐的靶向乐器

        song_3 = self._retrieve_song(target_3, personality_scores, pref_inst_3, exclude_ids)
        if song_3:
            exclude_ids.add(song_3['song_id'])
            playlist.append({
                "stage": f"第三阶段：目标达成 ({strategy_info['strategy']})",
                "track": song_3,
                "reason": strategy_info['description']
            })
        print("\n" + "=" * 65)
        print("🎵 心理学驱动的动态干预歌单 (Therapeutic Playlist)")
        print("=" * 65)
        for idx, item in enumerate(playlist):
            song = item['track']
            v, a = song['semantic_vector_28d'][0], song['semantic_vector_28d'][1]
            print(f"[{item['stage']}]")
            print(f"   ├─ 推荐理由: {item['reason']}")
            print(f"   ├─ 歌曲 ID : {song['song_id']}")
            print(f"   ├─ 真实 VA : V={v:.3f}, A={a:.3f}")
            # 展示 Top-5 的标签用于验证
            print(f"   ├─ 音乐标签: {song['display_tags'][:5]}")
            print(f"   └─ 试听路径: {song['song_path']}\n")

        return playlist


# ==========================================
# 本地测试模块
# ==========================================
if __name__ == "__main__":
    # 请确保路径指向你最新的 semantic_db.json
    DB_PATH = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\data\music_db\semantic_db.json"

    try:
        demo = PsychologyGuidedRec(DB_PATH)

        # 模拟情境：用户今天非常沮丧、焦虑甚至愤怒 (低 Valence, 高 Arousal)
        test_v_user = 0.2
        test_a_user = 0.8

        # 模拟用户的人格画像 (1-5分)：
        test_personality = {
            'O': 3,  # 开放性 (Openness) 中等
            'C': 4,  # 尽责性 (Conscientiousness) 偏高，算法会极力避免给他推荐重金属
            'E': 2,  # 外向性 (Extraversion) 偏低
            'A': 3,  # 宜人性 (Agreeableness) 中等
            'N': 4  # 神经质 (Neuroticism) 偏高，需要一定程度的发泄，但不能过火
        }

        # 生成歌单
        playlist = demo.generate_playlist(
            v_user=test_v_user,
            a_user=test_a_user,
            personality_scores=test_personality
        )

    except FileNotFoundError:
        print(f"❌ 找不到数据库文件: {DB_PATH}，请检查路径。")
    except Exception as e:
        print(f"❌ 发生错误: {e}")