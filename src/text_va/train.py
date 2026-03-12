import os
import sys
import warnings

# ============= 屏蔽警告以及其它打印的无用信息 ========================
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# 添加当前目录到 Python 路径，确保能导入同级模块
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
logging.getLogger("transformers").setLevel(logging.ERROR)
#
from data_loader import TextVADataLoader
from trainer import TextVATrainer
# 设置随机种子
torch.set_float32_matmul_precision('medium')

def setup_callbacks(cfg: DictConfig):
    callbacks = [
        TQDMProgressBar(refresh_rate=20),
        LearningRateMonitor(logging_interval='epoch')
    ]

    callbacks.append(EarlyStopping(
        monitor='val_loss', # 监控指标：验证集 Loss
        patience=3,         # 容忍度：3个 Epoch 不下降就停
        mode='min',         # 越小越好
        verbose=False
    ))

    ckpt_dir = Path(cfg.save.checkpoint_dir) / cfg.project.name
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best-epoch_{epoch:02d}-val_loss_{val_loss:.4f}',
        monitor='val_loss', # 监控指标
        mode='min',         # 越小越好
        save_top_k=1,       # 只保存最好的 1 个
        save_last=True,     # 同时保存最后一个 Epoch (用于断点续训)
        auto_insert_metric_name=False
    ))
    return callbacks

@hydra.main(version_base=None, config_path="../../configs", config_name="text_va")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.project.seed)
    print("=" * 60)
    print(f"🚀 Text Valence-Arousal Prediction [Chinese(CAVS+CAVT)+English(emobank+facebook)] ")
    print(f"📌 Model: {cfg.data.model_name}")
    print("=" * 60)

    print(">>> Initializing Data Module...")
    data_module = TextVADataLoader(cfg)
    data_module.setup()
    print(">>> Initializing Model System...")
    model_system = TextVATrainer(cfg)
    logger = TensorBoardLogger(save_dir=cfg.save.log_dir, name=cfg.project.name, version='')
    callbacks = setup_callbacks(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_model_summary=False,
        # 梯度裁剪，防止显存溢出或梯度爆炸 (NLP任务常用)
        gradient_clip_val=1.0 
    )
    # 开始训练
    print("\n>>> Start Training...")
    trainer.fit(model_system, datamodule=data_module)
    print("\n>>> Start Testing (Loading Best Model)...")
    trainer.test(model_system, datamodule=data_module, ckpt_path='best')
    print("\n✅ Training completed successfully!")
    print(f"Checkpoints saved to: {cfg.save.checkpoint_dir}/{cfg.project.name}")

if __name__ == "__main__":
    main()