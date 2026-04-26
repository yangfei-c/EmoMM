import os
import warnings
import logging
import torch
from transformers import AutoTokenizer
warnings.filterwarnings("ignore")
for _logger in ("transformers", "huggingface_hub", "urllib3", "filelock", "requests"): 
    logging.getLogger(_logger).setLevel(logging.ERROR)
try:
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except Exception:
    pass

try:
    from .trainer import TextVATrainer
except ImportError:
    from trainer import TextVATrainer


def load_model_and_tokenizer(ckpt_path, model_name="xlm-roberta-base"):
    """
    加载训练好的模型和分词器
    """
    # Informative prints are kept at a minimal level; transformers/hf internals are silenced
    # print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # print(f"Loading model from checkpoint: {ckpt_path}...")
    # 使用 PyTorch Lightning 的内置方法加载权重
    # 它会自动读取超参数并恢复模型架构
    model_system = TextVATrainer.load_from_checkpoint(ckpt_path)

    # 切换到评估模式，并将其移动到合适的设备 (GPU优先)
    model_system.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_system.to(device)

    return model_system, tokenizer, device


def predict_emotion(text, model_system, tokenizer, device, max_length=128):
    """
    对单条文本进行 Valence-Arousal 预测
    """
    # 1. 对文本进行 Tokenization (和 dataset.py 中保持完全一致)
    encoding = tokenizer(
        str(text),
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # 2. 将数据移动到设备上
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 3. 前向传播进行推理
    with torch.no_grad():  # 推理时不计算梯度
        logits = model_system(input_ids, attention_mask)

    # logits 的 shape 是 [1, 2]，输出已被 tanh 约束在 [-1, 1]
    valence = logits[0, 0].item()
    arousal = logits[0, 1].item()

    return valence, arousal


if __name__ == "__main__":
    # ==========================================
    # ⚠️ 请修改这里的 Checkpoint 路径！
    # 指向你在 ../checkpoints/Text_VA_Prediction/ 下最好的 .ckpt 文件
    # ==========================================
    CKPT_PATH = r"D:\AAAAAAAAAA_emotion\Code\MusicRecsys\MusicRecsys\src\text\checkpoints\Text_VA_Prediction\best-epoch_04-val_loss_0.2260.ckpt"
    MODEL_NAME = "xlm-roberta-base"

    # 检查权重文件是否存在
    if not os.path.exists(CKPT_PATH):
        print(f"❌ 找不到权重文件: {CKPT_PATH}")
        print("请检查路径是否正确，或者是否已经完成了训练。")
        exit()

    # 加载系统
    model_system, tokenizer, device = load_model_and_tokenizer(CKPT_PATH, MODEL_NAME)
    print(f"\n✅ 模型加载成功！使用设备: {device}\n")
    print("-" * 50)

    # 准备我们要测试的中英文句子
    test_texts = [
        "今天天气真好，和朋友一起去郊游，心情非常舒畅！",
        "The weather was so nice today. I went on a picnic with my friends and it was a really pleasant experience!",
        "I am so angry and frustrated with this terrible service!",  # 预期：低 Valence (负面), 高 Arousal (激动)
        "夜深了，听着窗外的淅沥雨声，感觉有些孤独和忧郁。",  # 预期：低 Valence (负面), 低 Arousal (平静/低沉)
        "I just won the lottery! This is the best day of my life!",  # 预期：极高 Valence, 极高 Arousal
        "会议好无聊，我想睡觉了。",  # 预期：中低 Valence, 极低 Arousal
    ]

    # 执行推理
    for text in test_texts:
        v, a = predict_emotion(text, model_system, tokenizer, device)
        print(f"📝 文本: {text}")
        print(f"📊 预测结果 -> Valence (效价/正负面): {v:.4f} | Arousal (唤醒度/激动平静): {a:.4f}")
        print("-" * 50)