import pytorch_lightning as pl
from torch.utils.data import DataLoader
#导入dataset
from dataset import JamendoDataset

class JamendoLoader(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return
        self.train_dataset = JamendoDataset(self.cfg, split='train')
        self.val_dataset = JamendoDataset(self.cfg, split='validation')
        self.test_dataset = JamendoDataset(self.cfg, split='test')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.data.batch_size,
            shuffle=True, num_workers=self.cfg.data.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.cfg.data.batch_size,
            shuffle=False, num_workers=self.cfg.data.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.cfg.data.batch_size,
            shuffle=False, num_workers=self.cfg.data.num_workers, pin_memory=True
        )