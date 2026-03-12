import logging
import os
import sys
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", message="The verbose parameter is deprecated")
warnings.filterwarnings("ignore", message="No positive samples in targets")
warnings.filterwarnings("ignore", ".*Consider setting `persistent_workers=True`.*")
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import torch

torch.set_float32_matmul_precision('medium')

from data_loader import JamendoLoader
from trainer import MusicTagTrainer


def setup_callbacks(cfg: DictConfig):
    callbacks = [
        TQDMProgressBar(refresh_rate=20),
        LearningRateMonitor(logging_interval='epoch')
    ]

    if 'early_stopping' in cfg.training:
        es = cfg.training.early_stopping
        callbacks.append(EarlyStopping(
            monitor=es.monitor, patience=es.patience, mode=es.mode, verbose=False
        ))

    ckpt_dir = Path(cfg.save.checkpoint_dir) / cfg.project_name/cfg.task_name
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='epoch_{epoch:02d}-val_loss_{val_loss:.4f}',
        monitor='val_loss', mode='min', save_top_k=1, save_last=True, auto_insert_metric_name=False
    ))
    return callbacks


@hydra.main(version_base=None, config_path="../../configs", config_name="music_tag")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    print("=" * 60)
    print(f"Music Tagging System")
    print(f"Task: {cfg.task_name} | Model: {cfg.model_type} | Loss: {cfg.loss_type}")
    print("=" * 60)

    print(">>> Initializing Data Module...")
    data_loader = JamendoLoader(cfg)
    data_loader.setup()


    tag_trainer = MusicTagTrainer(cfg)
    logger = TensorBoardLogger(save_dir=cfg.save.log_dir, name=cfg.project_name, version=cfg.task_name)
    callbacks = setup_callbacks(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_model_summary=True
    )

    print("\n>>> Start Training...")
    trainer.fit(tag_trainer, datamodule=data_loader)
    print("\n>>> Start Testing (Loading Best Model)...")
    trainer.test(tag_trainer, datamodule=data_loader, ckpt_path='best')
    print("\n✅ Training completed successfully!")
    print(f"Results saved to: {cfg.save.result_dir}/{cfg.project_name}/{cfg.task_name}")


if __name__ == "__main__":
    main()