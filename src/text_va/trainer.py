from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoConfig
#导入定义
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

    def forward(self, input_ids, attention_mask, labels=None):
        # model.forward 内部已经经过了 Sigmoid，输出在 0-1 之间
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
        r_val = r_num / (r_den + 1e-8)
        return r_val

    def compute_ccc_loss(self, logits, labels):
        """计算 CCC Loss"""
        # 拆分 Valence (列0) 和 Arousal (列1)
        logits_v = logits[:, 0]
        logits_a = logits[:, 1]
        labels_v = labels[:, 0]
        labels_a = labels[:, 1]

        # --- CCC Valence ---
        p_v = self.pearsonr(logits_v, labels_v)
        std_pred_v = torch.std(logits_v)
        std_true_v = torch.std(labels_v)

        num_v = 2 * p_v * std_pred_v * std_true_v
        den_v = torch.var(logits_v) + torch.var(labels_v) + torch.square(torch.mean(logits_v) - torch.mean(labels_v))
        ccc_v = num_v / (den_v + 1e-8)
        loss_v = 1 - ccc_v

        # --- CCC Arousal ---
        p_a = self.pearsonr(logits_a, labels_a)
        std_pred_a = torch.std(logits_a)
        std_true_a = torch.std(labels_a)

        num_a = 2 * p_a * std_pred_a * std_true_a
        den_a = torch.var(logits_a) + torch.var(labels_a) + torch.square(torch.mean(logits_a) - torch.mean(labels_a))
        ccc_a = num_a / (den_a + 1e-8)
        loss_a = 1 - ccc_a

        total_loss = 0.5 * loss_v + 0.5 * loss_a

        return total_loss, ccc_v, ccc_a

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)
        loss, ccc_v, ccc_a = self.compute_ccc_loss(logits, labels)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_ccc_v', ccc_v, prog_bar=True)
        self.log('train_ccc_a', ccc_a, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)
        loss, ccc_v, ccc_a = self.compute_ccc_loss(logits, labels)

        mse = torch.nn.functional.mse_loss(logits, labels)
        mae = torch.nn.functional.l1_loss(logits, labels)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_ccc_v', ccc_v)
        self.log('val_ccc_a', ccc_a)
        self.log('val_mse', mse)
        self.log('val_mae', mae)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)

        # 计算指标用于日志显示
        loss, ccc_v, ccc_a = self.compute_ccc_loss(logits, labels)
        self.log('test_ccc_v', ccc_v)
        self.log('test_ccc_a', ccc_a)

        # 预测值和真实值
        # move to cpu to save memory
        self.test_results.append({
            'predictions': logits.cpu(),
            'targets': labels.cpu()
        })

        return loss

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