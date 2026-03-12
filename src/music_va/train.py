import os
import sys
import warnings

# 屏蔽系统警告
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pathlib import Path
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
#
from data_loader import MusicVADataLoader
from trainer import MusicVATrainer

torch.set_float32_matmul_precision('medium')
#回调设置
def setup_callbacks(cfg: DictConfig):
    callbacks = [
        TQDMProgressBar(refresh_rate=20),
        LearningRateMonitor(logging_interval='epoch')
    ]
    callbacks.append(EarlyStopping(
        monitor='val_loss', patience=5, mode='min', verbose=False
    ))
    ckpt_dir = Path(cfg.save.checkpoint_dir) / cfg.project.name
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best-epoch_{epoch:02d}-val_loss_{val_loss:.4f}',
        monitor='val_loss', mode='min', save_top_k=1, save_last=True, auto_insert_metric_name=False
    ))
    return callbacks


@hydra.main(version_base=None, config_path="../../configs", config_name="music_va")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.project.seed)#设置种子
    print("=" * 60)
    print(f"🎵 Music Valence-Arousal Prediction (DEAM + PMEmo)")
    print("=" * 60)

    print(">>> Initializing Data Module...")
    data_module = MusicVADataLoader(cfg)
    data_module.setup()

    print(">>> Initializing Model System...")
    model_system = MusicVATrainer(cfg)
    logger = TensorBoardLogger(save_dir=cfg.save.log_dir, name=cfg.project.name, version='')

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=logger,
        callbacks=setup_callbacks(cfg),
        log_every_n_steps=10,
        enable_model_summary=False
    )

    print("\n>>> Start Training...")
    trainer.fit(model_system, datamodule=data_module)

    print("\n>>> Start Testing (Loading Best Model)...")
    trainer.test(model_system, datamodule=data_module, ckpt_path='best')


if __name__ == "__main__":
    main()