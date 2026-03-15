import os
import sys
import torch
import warnings
from transformers import AutoTokenizer

# 屏蔽警告
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, src_dir)
# 补上下面这三行，把各个子模块的路径都显式加进去
sys.path.insert(0, os.path.join(src_dir, 'text_va'))
sys.path.insert(0, os.path.join(src_dir, 'music_tag'))
sys.path.insert(0, os.path.join(src_dir, 'music_va'))

from text_va.trainer import TextVATrainer
from static_match import PsychologyGuidedRec

t_VA_CKPT = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\outputs\checkpoints\TextVAPred\last.ckpt"
SEMANTIC_DB_PATH = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\data\music_db\semantic_db.json"
# 3. 文本预训练模型名称
t_MODEL_NAME = "xlm-roberta-base"


def load_system():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 60)
    print("正在初始化系统...")
    print("=" * 60)
    print(f"[*] 计算设备: {device}")

    print("[*] 正在加载 NLP 情绪模块 (TextVA)...")
    tokenizer = AutoTokenizer.from_pretrained(t_MODEL_NAME)
    text_va_model = TextVATrainer.load_from_checkpoint(t_VA_CKPT, map_location=device).to(device)
    text_va_model.eval()

    print("[*] 正在加载 28维语义库与心理学推荐引擎...")
    recommender = PsychologyGuidedRec(SEMANTIC_DB_PATH)

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


def main_interactive():
    text_va_model, tokenizer, recommender, device = load_system()

    while True:
        try:
            # 1. 获取用户输入
            user_input = input("\n请描述一下你此刻的心情 (输入 'q' 退出): ")
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("\n系统退出。祝你今天心情愉快！")
                break
            if not user_input.strip():
                continue

            print("\n" + "-" * 50)
            print("正在分析文本语义与情绪坐标...")
            v_user, a_user = predict_user_emotion(user_input, text_va_model, tokenizer, device)
            recommender.match(
                v_user=v_user,
                a_user=a_user,
                user_tag_pref=None,
                weights=(0.7, 0.3),  # 情绪打分占70%，风格占30%
                top_k=3
            )

        except KeyboardInterrupt:
            print("\n系统退出。")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main_interactive()