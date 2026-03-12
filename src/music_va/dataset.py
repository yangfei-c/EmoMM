import os
import torch
import numpy as np
from torch.utils.data import Dataset

class MusicVADataset(Dataset):
    def __init__(self, data_list, cfg):
        self.data = data_list
        self.feature_root = cfg.data.feature_root

    def _load_mert_feature(self, song_id, source):
        song_dir = os.path.join(self.feature_root, source, 'mert_30s', str(song_id))
        segment_files = [f for f in os.listdir(song_dir) if f.endswith(".npy")]
        #一首歌曲的有效片段集
        valid_segments = []
        for f in segment_files:
            feat_path = os.path.join(song_dir, f)
            try:
                feat = np.load(feat_path)#(1,12,768)
                feat = torch.from_numpy(feat).float()
                #持续做平均，(1,12,768)-->(12,768)-->(768)(12层加权取平均)
                while feat.dim() > 1:
                    feat = feat.mean(dim=0)
                valid_segments.append(feat)
            except Exception:
                continue
        # 堆叠所有切片: (768) -> [num_segments, 768],
        # 所以训练是所有片段直接取平均，还是任选几个片段平均，
        # 同时，最值代表情绪爆发
        feat_stack = torch.stack(valid_segments)
        feat_mean = feat_stack.mean(dim=0)
        feat_max = feat_stack.max(dim=0)[0]#(情绪爆发点)
        feat_cat = torch.cat([feat_mean, feat_max], dim=-1)#(768)+(768)-->1536
        return feat_cat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        song_id = item['song_id']
        source = item['source']
        valence = float(item['valence'])
        arousal = float(item['arousal'])

        music_features = self._load_mert_feature(song_id, source)
        labels = torch.tensor([valence, arousal], dtype=torch.float32)

        return {
            'features': music_features,  # [1536]
            'labels': labels,  # [2]
            'song_id': song_id,  # 保留 ID 方便后期推荐溯源
            'source': source,#deam还是pmemo
        }