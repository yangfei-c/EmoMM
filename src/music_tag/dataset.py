import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig


class JamendoDataset(Dataset):
    def __init__(self, config: DictConfig, split: str = 'train'):
        self.cfg = config
        self.split = split #train
        self.model_type = config.model_type
        self.max_seq_len = config.data.max_seq_len

        self.data = self._load_data()
        self.genre_to_idx, self.instrument_to_idx, self.mood_to_idx = self._load_tag_mappings()

        self.num_genres = len(self.genre_to_idx)
        self.num_instruments = len(self.instrument_to_idx)
        self.num_moods = len(self.mood_to_idx)
        self.num_classes = self.num_genres + self.num_instruments + self.num_moods
        print(f"[{split.upper()}] Samples: {len(self.data)} | Mode: {self.model_type}")

    def _load_data(self):
        split_dir = self.cfg.data.split_dir
        json_filename = f"{self.split}.json"
        json_path = os.path.join(split_dir, json_filename)
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_tag_mappings(self):
        target_path = self.cfg.data.tag_path
        tags = np.load(target_path)
        # print(tags)#[genre---rock……,]
        genre_to_idx = {tag.split('---')[1]: i for i, tag in enumerate(tags[:10])}#{"genre---ambient":0}
        instrument_to_idx = {tag.split('---')[1]: i for i, tag in enumerate(tags[10:17])}
        mood_to_idx = {tag.split('---')[1]: i for i, tag in enumerate(tags[17:])}
        return genre_to_idx, instrument_to_idx, mood_to_idx

    def _get_labels(self, item_dict):
        genre_label = torch.zeros(self.num_genres, dtype=torch.float32)
        instrument_label = torch.zeros(self.num_instruments, dtype=torch.float32)
        mood_label = torch.zeros(self.num_moods, dtype=torch.float32)

        for g in item_dict.get('genres', []):#json文件中寻找genres键
            if g in self.genre_to_idx:
                genre_label[self.genre_to_idx[g]] = 1.0#genre_label：[0,0,……0,0]---->[0,1……0,0]
        for i in item_dict.get('instruments', []):
            if i in self.instrument_to_idx:
                instrument_label[self.instrument_to_idx[i]] = 1.0
        for m in item_dict.get('moods/themes', []):
            if m in self.mood_to_idx:
                mood_label[self.mood_to_idx[m]] = 1.0
        return genre_label, instrument_label, mood_label

    # ================= [ 随机选取一个特征片段 ] =================
    def _load_mert_single(self, feat_dir):
        segments = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
        selected = random.choice(segments) if self.split == 'train' else segments[0]
        feat = np.load(os.path.join(feat_dir, selected))
        feat = torch.from_numpy(feat).float()
        if feat.dim() > 1: feat = feat.mean(dim=0)
        return feat
    # ================= [ 对一首音乐的所有特征npy进行排序按先后顺序 ] =================
    def _load_mert_seq(self, feat_dir):
        feat_seq = torch.zeros((self.max_seq_len, self.cfg.model.input_dim), dtype=torch.float32)
        mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        segment_files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
        segment_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        valid_segments = []
        for f in segment_files:
            feat = np.load(os.path.join(feat_dir, f))
            feat = torch.from_numpy(feat).float()
            while feat.dim() > 1: feat = feat.mean(dim=0)
            valid_segments.append(feat)

        num_valid = min(len(valid_segments), self.max_seq_len)
        if num_valid > 0:
            feat_seq[:num_valid, :] = torch.stack(valid_segments[:num_valid])
            mask[:num_valid] = False
        return feat_seq, mask

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        path_str = item['path'].replace('/', os.sep)
        feat_dir = os.path.join(self.cfg.data.feature_dir, path_str)
        if self.model_type == "Transformer":
            x, mask = self._load_mert_seq(feat_dir)
        elif self.model_type=="MLP":
            x = self._load_mert_single(feat_dir)
            mask = torch.zeros(1, dtype=torch.bool)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        genre_y, instrument_y, mood_y = self._get_labels(item)#编码

        return x, mask, {
            'genre': genre_y,
            'instrument': instrument_y,
            'mood': mood_y
        }

import hydra
@hydra.main(version_base=None, config_path="../../configs", config_name="music_tag")
def main(cfg):
    from pprint import pprint
    jamendo=JamendoDataset(cfg,"train")
    print(jamendo.genre_to_idx)
    pprint(jamendo.data[0])
    sample=jamendo[0]
    mask=sample[1]
    tag_label=sample[2]
    pprint(tag_label)
    print(f"mask:{mask}")
if __name__=="__main__":
    main()
