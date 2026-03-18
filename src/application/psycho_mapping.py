import numpy as np

GENRE_IDX = {
    "ambient": 2, "classical": 3, "electronic": 4, "folk": 5,
    "hiphop": 6, "jazz": 7, "metal": 8, "pop": 9,
    "rock": 10, "soundtrack": 11
}

INST_IDX = {
    "bass": 12, "drums": 13, "guitar": 14, "piano": 15,
    "strings": 16, "synthesizer": 17, "voice": 18
}

MOOD_IDX = {
    "dark": 19, "dream": 20, "energetic": 21, "epic": 22,
    "film": 23, "happy": 24, "love": 25, "relaxing": 26, "sad": 27
}

INSTRUMENT_EFFECTS = {
    "strings": {  # 对应原理论中的小提琴/弦乐
        "role": "solace",  # 慰藉/共情
        "v_shift": -0.2,  # 倾向于压低愉悦度（更悲伤）
        "a_shift": 0.0,
        "description": "弦乐的音色能极大地提供共情与慰藉，适合极度悲伤时的情感接驳。"
    },
    "piano": {
        "role": "mental_work",  # 心理重构/深度反思
        "v_shift": 0.0,
        "a_shift": -0.3,  # 显著降低唤醒度
        "description": "纯净的钢琴声有助于引导深度反思，帮助大脑进入平静状态。"
    },
    "synthesizer": {
        "role": "distraction",  # 情感剥离/注意力转移
        "v_shift": 0.2,  # 强制拉升愉悦度
        "a_shift": 0.2,
        "description": "电子合成器的包络能剥离负面情绪的感染力，强制转移注意力。"
    },
    "guitar": {
        "role": "relaxation",  # 放松
        "v_shift": 0.1,
        "a_shift": -0.2,
        "description": "吉他清脆的质感质朴无攻击性，能有效缓解压力。"
    }
}

def calculate_penalty(personality, vector_28d):
    """
    personality: dict {'O':1-5, 'C':1-5, 'E':1-5, 'A':1-5, 'N':1-5}
    vector_28d: 歌曲的 28 维特征向量
    """
    score_modifier = 0.0

    p_metal = vector_28d[GENRE_IDX["metal"]]
    p_rock = vector_28d[GENRE_IDX["rock"]]
    p_classical = vector_28d[GENRE_IDX["classical"]]
    p_electronic = vector_28d[GENRE_IDX["electronic"]]
    p_pop = vector_28d[GENRE_IDX["pop"]]
    p_folk = vector_28d[GENRE_IDX["folk"]]
#大五人格
    #尽责性 (C): 偏好质朴(folk/pop)，极度反感强烈失真(metal/rock)
    if personality['C'] >= 4:
        score_modifier += p_metal * (-5.0)
        score_modifier += p_rock * (-2.0)
        score_modifier += p_folk * (+1.5)
    # 2. 开放性 (O): 喜欢复杂、有深度的音乐(classical)
    if personality['O'] >= 4:
        score_modifier += p_classical * (+2.0)
    # 3. 神经质 (N): 焦虑时偏好强烈音乐(metal/rock)发泄，反感过分柔和的音乐
    if personality['N'] >= 4:
        score_modifier += (p_metal + p_rock) * (+3.0)
        score_modifier += p_classical * (-1.0)
    # 4. 外向性 (E): 喜欢当代、有节奏感的音乐(electronic/pop)
    if personality['E'] >= 4:
        score_modifier += (p_electronic + p_pop) * (+2.0)
    elif personality['E'] <= 2:
        score_modifier += p_electronic * (-1.0)  # 内向者可能觉得太吵
    # 5. 宜人性 (A): 喜欢柔和无攻击性的音乐
    if personality['A'] >= 4:
        score_modifier += (p_pop + p_folk) * (+1.5)
    return score_modifier


# ==========================================
# 3. 动态情绪干预策略路由
# ==========================================
def determine_intervention_strategy(v_user, a_user, personality):
    v_threshold = 0.4
    a_threshold = 0.6

    # 高唤醒负面情绪 (愤怒、极度焦虑)
    if v_user < v_threshold and a_user > a_threshold:
        if personality['N'] >= 4:  # 神经质高 -> 发泄
            return {
                "strategy": "Discharge (宣泄)",
                "target_va": np.array([v_user, a_user]),
                "pref_instruments": ["synthesizer", "drums", "guitar"],  # 顺应高能
                "description": "顺应你的高能量情绪，通过强烈的节奏释放压力。"
            }
        else:  # 神经质低 -> 抑制与平复
            return {
                "strategy": "Catharsis (宣泄与平复)",
                "target_va": np.array([v_user, a_user - 0.3]),
                "pref_instruments": ["piano", "strings"],
                "description": "柔和的音色与平稳的节奏帮你逐渐平复内心的焦躁。"
            }

    #低唤醒负面情绪 (抑郁、疲惫、悲伤)
    elif v_user < v_threshold and a_user <= a_threshold:
        if personality['O'] >= 3:  # 开放性高 -> 共情与重构
            return {
                "strategy": "Solace & Mental Work (慰藉与心理重构)",
                "target_va": np.array([v_user, a_user]),
                "pref_instruments": ["strings", "piano"],  # 弦乐共情
                "description": "这首歌能懂你的失落，给你一个安全的空间去整理思绪。"
            }
        else:  # 开放性低 -> 强行转移注意力
            return {
                "strategy": "Mood Repair (心境修复)",
                "target_va": np.array([min(v_user + 0.4, 0.8), 0.5]),
                "pref_instruments": ["synthesizer", "pop"],  # 电子音色剥离负面情绪
                "description": "强行打断负面思绪，用合成器的律动帮你转移注意力。"
            }

    #正向情绪 (快乐、平静)
    else:
        if a_user < 0.5:  # 唤醒度低 -> 注入能量
            return {
                "strategy": "Energizing (注入能量)",
                "target_va": np.array([v_user, min(a_user + 0.4, 0.9)]),
                "pref_instruments": ["synthesizer", "guitar", "drums"],
                "description": "为你注入活力，让身体跟着节奏动起来！"
            }
        else:  # 高唤醒高愉悦 -> 积极强化
            return {
                "strategy": "Happy Maintenance (积极强化)",
                "target_va": np.array([min(v_user + 0.1, 1.0), a_user]),
                "pref_instruments": [],
                "description": "维持当前的快乐心境，享受此刻的美好！"
            }