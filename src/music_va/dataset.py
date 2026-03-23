import os
import torch
import numpy as np
from torch.utils.data import Dataset


class MusicVADataset(Dataset):
    def __init__(self, data_list, cfg):
        self.data = data_list
        self.feature_root = cfg.data.feature_root
        self.label_range = str(getattr(cfg.data, "label_range", "m11")).lower()

    def _load_mert_feature(self, song_id, source):
        song_dir = os.path.join(self.feature_root, source, 'mert_30s', str(song_id))
        segment_files = [f for f in os.listdir(song_dir) if f.endswith(".npy")]
        valid_segments = []
        for f in segment_files:
            feat_path = os.path.join(song_dir, f)
            try:
                feat = np.load(feat_path)
                feat = torch.from_numpy(feat).float()
                while feat.dim() > 1:
                    feat = feat.mean(dim=0)
                valid_segments.append(feat)
            except Exception:
                continue

        if not valid_segments:
            raise ValueError(f"No valid MERT segments found for song_id={song_id}, source={source}")

        feat_stack = torch.stack(valid_segments)
        feat_mean = feat_stack.mean(dim=0)
        feat_max = feat_stack.max(dim=0)[0]
        feat_cat = torch.cat([feat_mean, feat_max], dim=-1)
        return feat_cat

    def _map_label_range(self, valence: float, arousal: float):
        if self.label_range == "m11":
            return 2.0 * valence - 1.0, 2.0 * arousal - 1.0
        return valence, arousal

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        song_id = item['song_id']
        source = item['source']
        valence = float(item['valence'])
        arousal = float(item['arousal'])
        valence, arousal = self._map_label_range(valence, arousal)

        music_features = self._load_mert_feature(song_id, source)
        labels = torch.tensor([valence, arousal], dtype=torch.float32)

        return {
            'features': music_features,
            'labels': labels,
            'song_id': song_id,
            'source': source,
        }