from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoConfig

try:
    from .model import XLMRobertaVA
except ImportError:
    from model import XLMRobertaVA


class TextVATrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # 加载配置和模型
        config = AutoConfig.from_pretrained(cfg.data.model_name, num_labels=2)
        self.model = XLMRobertaVA.from_pretrained(cfg.data.model_name, config=config)
        # 保存测试结果
        self.test_results = []

        loss_cfg = getattr(cfg, "loss", None)
        self.w_mse = float(getattr(loss_cfg, "w_mse", 0.3)) if loss_cfg is not None else 0.3
        self.w_ccc = float(getattr(loss_cfg, "w_ccc", 0.7)) if loss_cfg is not None else 0.7

    def forward(self, input_ids, attention_mask, labels=None):
        # model.forward 内部经过 tanh，输出在 [-1,1]
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits

    def pearsonr(self, x, y):
        """计算皮尔逊相关系数"""
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        xm = x.sub(mean_x)
        ym = y.sub(mean_y)
        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        return r_num / (r_den + 1e-8)

    def compute_ccc_components(self, preds, targets):
        """计算 CCC 组件"""
        corr = self.pearsonr(preds, targets)
        std_pred = torch.std(preds, unbiased=False)
        std_true = torch.std(targets, unbiased=False)
        var_pred = torch.var(preds, unbiased=False)
        var_true = torch.var(targets, unbiased=False)
        mean_gap2 = torch.square(torch.mean(preds) - torch.mean(targets))
        ccc = (2.0 * corr * std_pred * std_true) / (var_pred + var_true + mean_gap2 + 1e-8)
        return ccc

    def compute_losses(self, logits, labels):
        """计算损失"""
        ccc_v = self.compute_ccc_components(logits[:, 0], labels[:, 0])
        ccc_a = self.compute_ccc_components(logits[:, 1], labels[:, 1])
        ccc_mean = 0.5 * (ccc_v + ccc_a)

        ccc_loss = 1.0 - ccc_mean
        mse_loss = torch.nn.functional.mse_loss(logits, labels)
        total_loss = self.w_ccc * ccc_loss + self.w_mse * mse_loss

        return {
            "total": total_loss,
            "ccc_loss": ccc_loss,
            "mse": mse_loss,
            "ccc_v": ccc_v,
            "ccc_a": ccc_a,
            "ccc_mean": ccc_mean,
        }

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)
        metrics = self.compute_losses(logits, labels)

        self.log('train_loss', metrics['total'], prog_bar=True)
        self.log('train_mse', metrics['mse'])
        self.log('train_ccc', metrics['ccc_mean'], prog_bar=True)
        self.log('train_ccc_v', metrics['ccc_v'])
        self.log('train_ccc_a', metrics['ccc_a'])
        return metrics['total']

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)
        metrics = self.compute_losses(logits, labels)
        mae = torch.nn.functional.l1_loss(logits, labels)

        self.log('val_loss', metrics['total'], prog_bar=True)
        self.log('val_mse', metrics['mse'])
        self.log('val_mae', mae)
        self.log('val_ccc', metrics['ccc_mean'], prog_bar=True)
        self.log('val_ccc_v', metrics['ccc_v'])
        self.log('val_ccc_a', metrics['ccc_a'])
        return metrics['total']

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)
        metrics = self.compute_losses(logits, labels)

        self.log('test_loss', metrics['total'])
        self.log('test_mse', metrics['mse'])
        self.log('test_ccc', metrics['ccc_mean'])
        self.log('test_ccc_v', metrics['ccc_v'])
        self.log('test_ccc_a', metrics['ccc_a'])

        self.test_results.append({
            'predictions': logits.cpu(),
            'targets': labels.cpu()
        })
        return metrics['total']

    def on_test_epoch_end(self):
        """
        保存结果到 results.pt
        """
        if self.test_results:
            #拼接所有 Batch 的结果
            all_preds = torch.cat([x['predictions'] for x in self.test_results])
            all_targets = torch.cat([x['targets'] for x in self.test_results])

            result_dir = Path(self.cfg.save.result_dir) / self.cfg.project.name
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_dir / "results.pt"
            torch.save({
                'predictions': all_preds,  # 形状: [N, 2] (Valence, Arousal)
                'targets': all_targets,  # 形状: [N, 2]
            }, result_path)

            print(f"\n[Result] 结果已保存至: {result_path}")
            print(f"         包含数据: predictions (预测值), targets (真实值)")

        # 清空列表以释放内存
        self.test_results.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=self.cfg.model.learning_rate,
                          weight_decay=self.cfg.model.weight_decay)

        if self.trainer.max_epochs:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = 1000

        warmup_steps = int(total_steps * 0.1)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }