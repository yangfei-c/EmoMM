import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import gradio as gr
import pandas as pd
import datetime
from text_app import load_system, predict_user_emotion

text_va_model, tokenizer, recommender, device = load_system()


def process_user_input(user_text):
    v_user, a_user = predict_user_emotion(user_text, text_va_model, tokenizer, device)

    ours_songs = recommender.match(v_user, a_user, weights=(0.7, 0.3), top_k=1)
    ours_song = ours_songs[0]

    diagnosis = f"系统诊断你的当前情绪 -> 效价(V): {v_user:.2f}, 唤醒度(H): {a_user:.2f}"

    return diagnosis, ours_song['song_path'], ours_song['song_path']


def save_feedback(text, score1, score2, preference):

    log_file = "user_study_results.csv"

    data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": text,
        "healing_score_song_A": score1,
        "healing_score_song_B": score2,
        "prefer": preference
    }

    df = pd.DataFrame([data])

    if not os.path.exists(log_file):
        df.to_csv(log_file, index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(log_file, index=False, mode='a', header=False, encoding='utf-8-sig')

    return "感谢参与！你的数据已记录！"


custom_css = """
/* 提高音频容器的最小高度，避免裁剪 */
.gradio-container .gr-audio {
    min-height: 220px !important;
}

/* 让波形区域有足够底部间距 */
.gradio-container .waveform-container {
    margin-bottom: 20px !important;
}

/* 确保控制条区域不被隐藏 */
.gradio-container .audio-controls {
    overflow: visible !important;
    height: auto !important;
}
"""


with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as web_ui:

    gr.Markdown("# 🎵 音乐辅助情绪调节系统")
    gr.Markdown("请写下你此刻的心情，系统将推荐两首音乐。请戴上耳机试听，并根据你的真实感受打分。")

    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            user_input = gr.Textbox(
                lines=3,
                placeholder="例如：今天被老板骂了，好烦啊... / e.g., I'm feeling so frustrated today...",
                label="在此输入你的心情 (中/英)"
            )

            submit_btn = gr.Button("开始分析并匹配适合音乐", variant="primary")

            diagnosis_out = gr.Textbox(label="情绪分析", interactive=False)

        with gr.Column(scale=3):

            gr.Markdown("### 🎧 盲测试听区")

            with gr.Row(equal_height=False):

                with gr.Column(min_width=350):

                    gr.Markdown("#### 推荐曲目 A")

                    audio_a = gr.Audio(
                        label="点击播放",
                        type="filepath"
                    )

                    score_a = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        label="这首歌对你心情的治愈程度 (1-5分)"
                    )

                with gr.Column():

                    gr.Markdown("#### 推荐曲目 B")

                    audio_b = gr.Audio(
                        label="点击播放",
                        type="filepath"
                    )

                    score_b = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        label="这首歌对你心情的治愈程度 (1-5分)"
                    )

            preference = gr.Radio(
                choices=["我更喜欢 A", "我更喜欢 B", "都差不多"],
                label="总体偏好"
            )

            feedback_btn = gr.Button("提交我的打分")

            feedback_out = gr.Textbox(label="提交状态", interactive=False)


    submit_btn.click(
        fn=process_user_input,
        inputs=[user_input],
        outputs=[diagnosis_out, audio_a, audio_b]
    )

    feedback_btn.click(
        fn=save_feedback,
        inputs=[user_input, score_a, score_b, preference],
        outputs=[feedback_out]
    )


if __name__ == "__main__":

    AUDIO_ROOT_DIR = "D:/AAAA_audio_dataset"

    web_ui.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[AUDIO_ROOT_DIR]
    )