import json
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
#
from dataset import MusicVADataset


class MusicVADataLoader(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        print(">>> Loading Music VA JSON Data...")
        json_path = self.cfg.data.json_path
        with open(json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        total_len = len(all_data)
        print(f"    Total (DEAM+PMEmo): {total_len}")

        # 随机划分 (Train / Val / Test)
        val_len = int(total_len * self.cfg.training.val_ratio)
        test_len = int(total_len * self.cfg.training.test_ratio)
        train_len = total_len - val_len - test_len

        generator = torch.Generator().manual_seed(self.cfg.project.seed)
        all_indices = torch.randperm(total_len, generator=generator).tolist()
        train_indices = all_indices[:train_len]
        val_indices = all_indices[train_len: train_len + val_len]
        test_indices = all_indices[train_len + val_len:]

        train_data = [all_data[i] for i in train_indices]
        val_data = [all_data[i] for i in val_indices]
        test_data = [all_data[i] for i in test_indices]
        print(f"    Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

        # 实例化 Dataset
        self.train_dataset = MusicVADataset(train_data, self.cfg)
        self.val_dataset = MusicVADataset(val_data, self.cfg)
        self.test_dataset = MusicVADataset(test_data, self.cfg)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )