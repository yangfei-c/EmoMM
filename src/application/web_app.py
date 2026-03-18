import os
import sys
import torch
import warnings
import datetime
import re
import pandas as pd
import gradio as gr
from transformers import AutoTokenizer

# 屏蔽警告
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

# ==========================================
# 动态加载模块路径
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.join(src_dir, 'text_va'))
sys.path.insert(0, os.path.join(src_dir, 'music_tag'))
sys.path.insert(0, os.path.join(src_dir, 'music_va'))

from text_va.trainer import TextVATrainer
from static_match import PsychologyGuidedRec

# ==========================================
#局路径与常量配置
# ==========================================
t_MODEL_NAME = "xlm-roberta-base"
t_VA_CKPT = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\outputs\checkpoints\TextVAPred\last.ckpt"
SEMANTIC_DB_PATH = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\data\music_db\semantic_db.json"
AUDIO_ROOT_DIR = "D:/AAAA_audio_dataset"

def load_system():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 计算设备: {device}")

    print("[*] 正在加载 NLP 情绪模块 (TextVA)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(t_MODEL_NAME)
        text_va_model = TextVATrainer.load_from_checkpoint(t_VA_CKPT, map_location=device).to(device)
        text_va_model.eval()
    except Exception as e:
        print(f"[!] 模型加载失败: {e}")
        return None, None, None, device

    try:
        recommender = PsychologyGuidedRec(SEMANTIC_DB_PATH)
    except Exception as e:
        print(f"[!] 推荐器加载失败: {e}")
        recommender = None

    return text_va_model, tokenizer, recommender, device


def predict_user_emotion(text, model, tokenizer, device):
    enc = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))

    valence = logits[0, 0].item()
    arousal = logits[0, 1].item()
    return valence, arousal


text_va_model, tokenizer, recommender, device = load_system()


def clear_ui():
    return (
        " **正在深入分析您的情绪并制定干预策略，请稍候...**",
        gr.update(value=None), "正在检索匹配曲目...",
        gr.update(value=None), "正在检索匹配曲目...",
        gr.update(value=None), "正在检索匹配曲目..."
    )

def process_user_input(user_text, o_score, c_score, e_score, a_score, n_score):
    if text_va_model is None or tokenizer is None or recommender is None:
        return ("系统组件加载失败，请检查后台日志。", None, "", None, "", None, "")

    if not user_text.strip():
        return ("请先描述您此刻的心情！", None, "", None, "", None, "")

    v_user, a_user = predict_user_emotion(user_text, text_va_model, tokenizer, device)
    personality = {
        'O': o_score, 'C': c_score, 'E': e_score, 'A': a_score, 'N': n_score
    }

    playlist = recommender.generate_playlist(v_user, a_user, personality)

    while len(playlist) < 3:
        playlist.append({
            'track': {
                'song_path': None,
                'semantic_vector_28d': [0.0, 0.0],
                'display_tags': []
            },
            'reason': 'No suitable track found.',
            'stage': 'N/A'
        })

    stage_full = playlist[2].get('stage', '')
    match = re.search(r'\(([^)]+)\)', stage_full)
    strategy = match.group(1) if match else stage_full

    diagnosis_text = (
        f"### 情绪诊断\n"
        f"- **效价 (Valence / 愉悦度)**: `{v_user:.2f}`\n"
        f"- **唤醒度 (Arousal / 激动度)**: `{a_user:.2f}`\n\n"
        f"### 制定干预策略\n"
        f"- **核心策略**: **{strategy}**"
    )

    audio_paths = []
    info_texts = []

    for item in playlist[:3]:  # 只取前三个
        song = item['track']
        audio_paths.append(song.get('song_path'))  # 可能为 None

        v, a = song.get('semantic_vector_28d', [0.0, 0.0])[:2]
        tags = ", ".join([f"`{tag}`" for tag in song.get('display_tags', [])[:5]])

        info = (
            f"**疗愈理由**: {item.get('reason', '')}\n\n"
            f"**音乐标签**: {tags}\n\n"
            f"**真实坐标**: Valence={v:.2f}, Arousal={a:.2f}"
        )
        info_texts.append(info)

    # 返回七个值（使用 gr.update 强制播放位置归零）
    return (
        diagnosis_text,
        gr.update(value=audio_paths[0], playback_position=0.0), info_texts[0],
        gr.update(value=audio_paths[1], playback_position=0.0), info_texts[1],
        gr.update(value=audio_paths[2], playback_position=0.0), info_texts[2]
    )

def save_feedback(text, o_score, c_score, e_score, a_score, n_score, improvement_score):
    log_file = "user_study_results.csv"
    data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": text,
        "O": o_score, "C": c_score, "E": e_score, "A": a_score, "N": n_score,
        "improvement_score": improvement_score
    }
    df = pd.DataFrame([data])
    if not os.path.exists(log_file):
        df.to_csv(log_file, index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(log_file, index=False, mode='a', header=False, encoding='utf-8-sig')
    return "✅ 感谢参与！您的疗效数据已记录，这对我们的研究非常重要！"


custom_css = """
/* ---- 修复音频波形遮挡控制条的问题 ---- */
.gradio-container .gr-audio { 
    min-height: 220px !important; 
}
.gradio-container .waveform-container {
    margin-bottom: 20px !important;
}
.gradio-container .audio-controls {
    overflow: visible !important;
    height: auto !important;
}
/* ---- UI 美化样式 ---- */
.section-header { 
    font-size: 1.2em; 
    font-weight: bold; 
    color: #2c3e50; 
    margin-bottom: 10px; 
    border-bottom: 2px solid #eee; 
    padding-bottom: 5px; 
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as web_ui:
    gr.Markdown("# 🎵 情感感知与心理学驱动的动态音乐干预系统")
    gr.Markdown("*(Psychology-Guided Dynamic Music Intervention System based on Continuous Emotion Tracking)*")

    with gr.Row(equal_height=False):
        # ================= 左侧：问诊区 =================
        with gr.Column(scale=1, min_width=350):
            gr.Markdown("<div class='section-header'>心理与情绪问诊</div>")

            user_input = gr.Textbox(
                lines=4,
                placeholder="例如：今天被老板骂了，好烦啊... 或者... 刚跑完步，感觉很有活力！",
                label="请描述您此刻的遭遇与心情 (中/英文皆可)"
            )

            gr.Markdown("####请评估您的性格特质 (1=极不符合, 5=极其符合)")
            with gr.Group():
                o_score = gr.Slider(1, 5, value=3, step=1, label="我喜欢探索抽象复杂的艺术 (开放性 Openness)")
                c_score = gr.Slider(1, 5, value=3, step=1, label="我做事严谨有条理，反感混乱 (尽责性 Conscientiousness)")
                e_score = gr.Slider(1, 5, value=3, step=1, label="我喜欢参加热闹的聚会 (外向性 Extraversion)")
                a_score = gr.Slider(1, 5, value=3, step=1, label="我富有同情心，容易共情他人 (宜人性 Agreeableness)")
                n_score = gr.Slider(1, 5, value=3, step=1, label="面对压力时，我很容易陷入焦虑 (神经质 Neuroticism)")

            submit_btn = gr.Button("开始心理诊断与歌单生成", variant="primary", size="lg")

            gr.Markdown("<br><div class='section-header'>系统诊断报告</div>")
            with gr.Column(variant="panel"):
                diagnosis_out = gr.Markdown("等待分析...")

        # ================= 右侧：处方歌单区 =================
        with gr.Column(scale=2):
            gr.Markdown("<div class='section-header'>ISO 原则干预歌单 (Therapeutic Playlist)</div>")
            gr.Markdown("请**按顺序**佩戴耳机试听以下三首曲目，体验情绪的平滑过渡。")

            with gr.Column(variant="panel"):
                gr.Markdown("### Track 1: 共情与接纳 (Empathy & Solace)")
                with gr.Row():
                    audio_1 = gr.Audio(label="点击播放", type="filepath", scale=1, elem_id="audio_1")
                    with gr.Column(scale=3):
                        info_1 = gr.Markdown("等待生成...")

            with gr.Column(variant="panel"):
                gr.Markdown("### Track 2: 过渡与重构 (Transition & Reappraisal)")
                with gr.Row():
                    audio_2 = gr.Audio(label="点击播放", type="filepath", scale=1, elem_id="audio_2")
                    with gr.Column(scale=3):
                        info_2 = gr.Markdown("等待生成...")

            with gr.Column(variant="panel"):
                gr.Markdown("###Track 3: 目标心境 (Target State)")
                with gr.Row():
                    audio_3 = gr.Audio(label="点击播放", type="filepath", scale=1, elem_id="audio_3")
                    with gr.Column(scale=3):
                        info_3 = gr.Markdown("等待生成...")

    # ================= 底部：反馈区 =================
    gr.Markdown("<br><div class='section-header'>疗效评估 (User Study)</div>")
    with gr.Row():
        improvement_score = gr.Slider(
            minimum=1, maximum=5, step=1, value=3,
            label="完整听完三首曲目后，您觉得自己的负面情绪改善程度？(1=毫无改善甚至更糟, 5=极大改善/被治愈)"
        )
        feedback_btn = gr.Button("提交疗效数据", variant="secondary")

    feedback_out = gr.Textbox(label="提交状态", interactive=False)

    # ================= 绑定事件（零 JS） =================
    submit_btn.click(
        fn=clear_ui,
        inputs=[],
        outputs=[diagnosis_out, audio_1, info_1, audio_2, info_2, audio_3, info_3],
        queue=False
    ).then(
        fn=process_user_input,
        inputs=[user_input, o_score, c_score, e_score, a_score, n_score],
        outputs=[diagnosis_out, audio_1, info_1, audio_2, info_2, audio_3, info_3]
    )

    feedback_btn.click(
        fn=save_feedback,
        inputs=[user_input, o_score, c_score, e_score, a_score, n_score, improvement_score],
        outputs=[feedback_out]
    )

if __name__ == "__main__":
    web_ui.queue().launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[AUDIO_ROOT_DIR]
    )