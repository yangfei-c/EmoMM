import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TextVADataset(Dataset):
    def __init__(self, data_list, cfg):
        self.data = data_list
        self.max_len = cfg.data.max_length
        self.label_range = str(getattr(cfg.data, "label_range", "m11")).lower()
        # 加载分词器 (XLM-RoBERTa-base)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.data.model_name)

    def __len__(self):
        return len(self.data)

    def _map_label_range(self, valence: float, arousal: float):
        if self.label_range == "m11":
            return 2.0 * valence - 1.0, 2.0 * arousal - 1.0
        return valence, arousal

    def __getitem__(self, idx):
        item = self.data[idx]

        text = str(item['text'])
        # 文本处理：Tokenization
        # padding='max_length': 补齐到固定长度
        # truncation=True: 超长截断
        # return_tensors='pt': 返回 PyTorch Tensor
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # 标签处理：Valence 和 Arousal
        valence = float(item['valence'])
        arousal = float(item['arousal'])
        valence, arousal = self._map_label_range(valence, arousal)
        labels = torch.tensor([valence, arousal], dtype=torch.float)

        # squeeze(0) 是因为 tokenizer 默认返回 [1, len]，我们需要去掉 batch 维度变成 [len]
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }