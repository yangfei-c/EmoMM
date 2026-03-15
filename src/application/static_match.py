import json
import numpy as np
import time


class PsychologyGuidedRec:
    def __init__(self, db_path: str):
        print("[*] 正在加载语义音乐数据库...")
        start_time = time.time()
        with open(db_path, 'r', encoding='utf-8') as f:
            self.db = json.load(f)

        self.song_matrix = np.array([song['semantic_vector_28d'] for song in self.db])
        self.va_matrix = self.song_matrix[:, 0:2]  # 形状: (N, 2)
        self.tag_matrix = self.song_matrix[:, 2:28]  # 形状: (N, 26)
        print(f"[*] 加载完毕！共 {len(self.db)} 首歌。耗时: {time.time() - start_time:.2f} 秒")

    def _psychological_intervention(self, v_user, a_user):
        """
            《情绪一致性》与《心境修复》理论，计算目标检索坐标
        """
        print(f"\n用户当前状态: Valence={v_user:.2f}, Arousal={a_user:.2f}")

        # 宣泄调节 (Catharsis) - 针对高唤醒负面情绪 (如愤怒/极度焦虑)
        if v_user < 0.4 and a_user > 0.6:
            v_target = v_user  # 保持共情
            a_target = a_user - 0.2  # 引导平复
            strategy = "Catharsis (宣泄与共情)"

        # 心境修复 (Mood Repair) - 针对低唤醒负面情绪 (如疲惫/轻度抑郁)
        elif v_user < 0.4 and a_user <= 0.6:
            v_target = min(v_user + 0.3, 0.8)  # 强制拉升愉悦度
            a_target = 0.5
            strategy = "Mood Repair (正向转移)"

        # 积极强化 (Positive Reinforcement) - 针对积极情绪
        else:
            v_target = min(v_user + 0.1, 1.0)  # 锦上添花
            a_target = a_user
            strategy = "Positive Reinforcement (积极强化)"

        print(f"策略: {strategy}")
        print(f"偏移目标: Valence={v_target:.2f}, Arousal={a_target:.2f}")

        return np.array([v_target, a_target]), strategy

    def match(self, v_user, a_user, user_tag_pref=None, weights=(0.7, 0.3), top_k=5):

        target_va, strategy = self._psychological_intervention(v_user, a_user)

        va_distances = np.linalg.norm(self.va_matrix - target_va, axis=1)
        sigma = 0.5
        s_emotion = np.exp(- (va_distances ** 2) / (2 * sigma ** 2))

        if user_tag_pref is not None:
            user_tag_vec = np.array(user_tag_pref)
            norm_db = np.linalg.norm(self.tag_matrix, axis=1) + 1e-8
            norm_user = np.linalg.norm(user_tag_vec) + 1e-8
            s_style = np.dot(self.tag_matrix, user_tag_vec) / (norm_db * norm_user)
        else:
            s_style = np.ones(len(self.db))

        total_scores = weights[0] * s_emotion + weights[1] * s_style

        top_indices = np.argsort(total_scores)[::-1][:top_k]

        print("\n" + "=" * 50)
        print("🎵 推荐结果 (Top-K)")
        print("=" * 50)
        results = []
        for rank, idx in enumerate(top_indices):
            song = self.db[idx]
            score = total_scores[idx]
            song_v, song_a = song['semantic_vector_28d'][0], song['semantic_vector_28d'][1]
            print(f"[{rank + 1}] ID: {song['song_id']} | 综合匹配度: {score:.3f}")
            print(f"    ├─ 真实 VA: V={song_v:.3f}, A={song_a:.3f}")
            print(f"    ├─ 音乐标签: {song['display_tags'][:5]}")
            print(f"    └─ 试听路径: {song['song_path']}")
            results.append(song)

        return results


if __name__ == "__main__":
    DB_PATH = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\data\music_db\semantic_db.json"

    demo = PsychologyGuidedRec(DB_PATH)

    user_v = 0.2
    user_a = 0.3

    demo.match(v_user=user_v, a_user=user_a, user_tag_pref=None, top_k=3)