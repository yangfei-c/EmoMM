import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
#
from va_model import MusicVAModel


class MusicVATrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = MusicVAModel(cfg)
        self.test_results = []

    def forward(self, x):
        return self.model(x)

    # CCC 逻辑
    def pearsonr(self, x, y):
        mean_x, mean_y = torch.mean(x), torch.mean(y)
        xm, ym = x.sub(mean_x), y.sub(mean_y)
        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        return r_num / (r_den + 1e-8)
    #CCC Loss
    def compute_ccc_loss(self, logits, labels):
        logits_v, logits_a = logits[:, 0], logits[:, 1]
        labels_v, labels_a = labels[:, 0], labels[:, 1]

        # CCC Valence
        p_v = self.pearsonr(logits_v, labels_v)
        num_v = 2 * p_v * torch.std(logits_v) * torch.std(labels_v)
        den_v = torch.var(logits_v) + torch.var(labels_v) + torch.square(torch.mean(logits_v) - torch.mean(labels_v))
        ccc_v = num_v / (den_v + 1e-8)
        loss_v = 1 - ccc_v

        # CCC Arousal
        p_a = self.pearsonr(logits_a, labels_a)
        num_a = 2 * p_a * torch.std(logits_a) * torch.std(labels_a)
        den_a = torch.var(logits_a) + torch.var(labels_a) + torch.square(torch.mean(logits_a) - torch.mean(labels_a))
        ccc_a = num_a / (den_a + 1e-8)
        loss_a = 1 - ccc_a

        total_loss = 0.5 * loss_v + 0.5 * loss_a
        return total_loss, ccc_v, ccc_a

    def training_step(self, batch, batch_idx):
        features, labels = batch['features'], batch['labels']
        logits = self(features)
        loss, ccc_v, ccc_a = self.compute_ccc_loss(logits, labels)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_ccc_v', ccc_v, prog_bar=True)
        self.log('train_ccc_a', ccc_a, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch['features'], batch['labels']
        logits = self(features)
        loss, ccc_v, ccc_a = self.compute_ccc_loss(logits, labels)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_ccc_v', ccc_v)
        self.log('val_ccc_a', ccc_a)
        return loss

    def test_step(self, batch, batch_idx):
        features, labels = batch['features'], batch['labels']
        logits = self(features)
        loss, ccc_v, ccc_a = self.compute_ccc_loss(logits, labels)

        self.log('test_ccc_v', ccc_v)
        self.log('test_ccc_a', ccc_a)

        # 保存预测结果和歌曲ID，推荐用！
        self.test_results.append({
            'predictions': logits.cpu(),
            'targets': labels.cpu(),
            'song_ids': batch['song_id']
        })
        return loss

    def on_test_epoch_end(self):
        if self.test_results:
            all_preds = torch.cat([x['predictions'] for x in self.test_results])
            all_targets = torch.cat([x['targets'] for x in self.test_results])

            all_song_ids = []
            for x in self.test_results:
                all_song_ids.extend(x['song_ids'])

            from pathlib import Path
            result_dir = Path(self.cfg.save.result_dir) / self.cfg.project.name
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_dir / "results.pt"

            torch.save({
                'predictions': all_preds,
                'targets': all_targets,
                'song_ids': all_song_ids
            }, result_path)

            print(f"\n[Result] 音乐 VA 坐标预测结果已保存至: {result_path}")

        self.test_results.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(),
                          lr=self.cfg.model.learning_rate,
                          weight_decay=self.cfg.model.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches if self.trainer.max_epochs else 1000
        warmup_steps = int(total_steps * 0.1)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}