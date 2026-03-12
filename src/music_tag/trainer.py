import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import AUROC
#导入模型
from tag_model import TagModel
'''
    参考论文：《Speech Mutil-label Emotion Recognition Using Asymmetric 
                Class Loss Function Based on Effective Samples》 
'''

def calculate_dynamic_gammas(tag_sample_counts, alpha=0.5, beta=5.0, delta=0.999):
    counts = torch.tensor(tag_sample_counts, dtype=torch.float32)
    E_n = (1.0 - torch.pow(delta, counts)) / (1.0 - delta)
    E_n_total = torch.sum(E_n)
    ratio = E_n / E_n_total
    gamma_pos = alpha * ratio
    gamma_neg = beta * (1.0 - ratio)
    return gamma_pos, gamma_neg

class DynamicAsymmetricClassLoss(nn.Module):
    def __init__(self, gamma_pos, gamma_neg, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        gamma_pos = self.gamma_pos.to(x.device).unsqueeze(0)
        gamma_neg = self.gamma_neg.to(x.device).unsqueeze(0)
        x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x_sigmoid.clamp(min=self.eps, max=1 - self.eps)
        pt = x_sigmoid * y + (1 - x_sigmoid) * (1 - y)
        one_sided_gamma = gamma_pos * y + gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        if self.clip > 0:
            one_sided_w = one_sided_w * (1 - y) * (x_sigmoid > self.clip).float() + one_sided_w * y
        loss = - one_sided_w * torch.log(pt)
        return loss.mean()
'''
    参考论文：《Asymmetric Loss For Multi-Label Classification》 
'''
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x_sigmoid.clamp(min=self.eps, max=1 - self.eps)
        pt = x_sigmoid * y + (1 - x_sigmoid) * (1 - y)
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        if self.clip > 0:
            one_sided_w = one_sided_w * (1 - y) * (x_sigmoid > self.clip).float() + one_sided_w * y
        loss = - one_sided_w * torch.log(pt)
        return loss.mean()

class MusicTagTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = TagModel(cfg)

        self.genre_weight = cfg.weights.genre
        self.instrument_weight = cfg.weights.instrument
        self.mood_weight = cfg.weights.mood

        loss_type = cfg.loss_type
        if loss_type == "BCE":
            self.genre_criterion = nn.BCEWithLogitsLoss()
            self.instrument_criterion = nn.BCEWithLogitsLoss()
            self.mood_criterion = nn.BCEWithLogitsLoss()
        elif loss_type == "ASL":
            self.genre_criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
            self.instrument_criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
            self.mood_criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
        elif loss_type == "DynamASCL":
            genre_counts = [5972, 4103, 13151, 1816, 1999, 1989, 1226, 5935, 5551, 5647]
            instrument_counts = [4773, 5054, 3952, 5947, 905, 5941, 1355]
            mood_counts = [857, 762, 1023, 659, 983, 1109, 732, 1000, 617]
            g_gamma_pos, g_gamma_neg = calculate_dynamic_gammas(genre_counts)
            i_gamma_pos, i_gamma_neg = calculate_dynamic_gammas(instrument_counts)
            m_gamma_pos, m_gamma_neg = calculate_dynamic_gammas(mood_counts)

            self.genre_criterion = DynamicAsymmetricClassLoss(gamma_pos=g_gamma_pos, gamma_neg=g_gamma_neg, clip=0.05)
            self.instrument_criterion = DynamicAsymmetricClassLoss(gamma_pos=i_gamma_pos, gamma_neg=i_gamma_neg, clip=0.05)
            self.mood_criterion = DynamicAsymmetricClassLoss(gamma_pos=m_gamma_pos, gamma_neg=m_gamma_neg, clip=0.05)
        else:
            raise ValueError(f"❌ 不支持的损失函数类型: {loss_type}")

        self.val_genre_auroc = AUROC(task="multilabel", num_labels=cfg.task.num_genres, average='macro')
        self.val_instrument_auroc = AUROC(task="multilabel", num_labels=cfg.task.num_instruments, average='macro')
        self.val_mood_auroc = AUROC(task="multilabel", num_labels=cfg.task.num_moods, average='macro')
        self.test_genre_auroc = AUROC(task="multilabel", num_labels=cfg.task.num_genres, average='macro')
        self.test_instrument_auroc = AUROC(task="multilabel", num_labels=cfg.task.num_instruments, average='macro')
        self.test_mood_auroc = AUROC(task="multilabel", num_labels=cfg.task.num_moods, average='macro')
        self.test_step_outputs = []

    def forward(self, x, mask=None):
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        x, mask, y_dict = batch
        outputs = self(x, mask)

        genre_loss = self.genre_criterion(outputs['genre'], y_dict['genre'])
        instrument_loss = self.instrument_criterion(outputs['instrument'], y_dict['instrument'])
        mood_loss = self.mood_criterion(outputs['mood'], y_dict['mood'])

        total_loss = self.genre_weight * genre_loss + self.instrument_weight * instrument_loss + self.mood_weight * mood_loss
        self.log('train_loss', total_loss, prog_bar=True)

        return total_loss
    def validation_step(self, batch, batch_idx):
        x, mask, y_dict = batch
        outputs = self(x, mask)

        genre_loss = self.genre_criterion(outputs['genre'], y_dict['genre'])
        instrument_loss = self.instrument_criterion(outputs['instrument'], y_dict['instrument'])
        mood_loss = self.mood_criterion(outputs['mood'], y_dict['mood'])
        total_val_loss = self.genre_weight * genre_loss + self.instrument_weight * instrument_loss + self.mood_weight * mood_loss

        self.val_genre_auroc(outputs['genre'], y_dict['genre'].long())
        self.val_instrument_auroc(outputs['instrument'], y_dict['instrument'].long())
        self.val_mood_auroc(outputs['mood'], y_dict['mood'].long())

        self.log('val_loss', total_val_loss, prog_bar=True)
        self.log('val_genre_auroc', self.val_genre_auroc, on_epoch=True)
        self.log('val_instrument_auroc', self.val_instrument_auroc, on_epoch=True)
        self.log('val_mood_auroc', self.val_mood_auroc, on_epoch=True)
        return total_val_loss

    def test_step(self, batch, batch_idx):
        x, mask, y_dict = batch
        outputs = self(x, mask)

        self.test_genre_auroc(outputs['genre'], y_dict['genre'].long())
        self.test_instrument_auroc(outputs['instrument'], y_dict['instrument'].long())
        self.test_mood_auroc(outputs['mood'], y_dict['mood'].long())

        res_dict = {
            'genre_pred': outputs['genre'].cpu(),
            'instrument_pred': outputs['instrument'].cpu(),
            'mood_pred': outputs['mood'].cpu(),

            'genre_target': y_dict['genre'].cpu(),
            'instrument_target': y_dict['instrument'].cpu(),
            'mood_target': y_dict['mood'].cpu(),

            'features': outputs['features'].cpu(),
        }
        # 如果架构是 AttenSeq，则保存提取出的注意力权重
        if outputs.get('attn_weights') is not None:
            res_dict['attn_weights'] = outputs['attn_weights'].cpu()

        self.test_step_outputs.append(res_dict)

        self.log('test_genre_auroc', self.test_genre_auroc)
        self.log('test_instrument_auroc', self.test_instrument_auroc)
        self.log('test_mood_auroc', self.test_mood_auroc)

    def on_test_epoch_end(self):
        if self.test_step_outputs:
            all_genre_pred = torch.cat([x['genre_pred'] for x in self.test_step_outputs])
            all_instrument_pred = torch.cat([x['instrument_pred'] for x in self.test_step_outputs])
            all_mood_pred = torch.cat([x['mood_pred'] for x in self.test_step_outputs])

            all_genre_target = torch.cat([x['genre_target'] for x in self.test_step_outputs])
            all_instrument_target = torch.cat([x['instrument_target'] for x in self.test_step_outputs])
            all_mood_target = torch.cat([x['mood_target'] for x in self.test_step_outputs])

            all_features = torch.cat([x['features'] for x in self.test_step_outputs])

            torch_save_dict = {
                'predictions': torch.cat([all_genre_pred, all_instrument_pred, all_mood_pred], dim=1),
                'targets': torch.cat([all_genre_target, all_instrument_target, all_mood_target], dim=1),
                'genre_predictions': all_genre_pred,
                'instrument_predictions': all_instrument_pred,
                'mood_predictions': all_mood_pred,
                'features': all_features,
            }

            # 安全地整合注意力权重 (如果存在)
            if 'attn_weights' in self.test_step_outputs[0]:
                all_attn_weights = torch.cat([x['attn_weights'] for x in self.test_step_outputs])
                torch_save_dict['attn_weights'] = all_attn_weights

            from pathlib import Path
            result_dir = Path(self.cfg.save.result_dir) / self.cfg.project_name /self.cfg.task_name
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_dir / "results.pt"

            torch.save(torch_save_dict, result_path)
            print(f"\n[Result] 测试结果已保存至: {result_path}")

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.model.learning_rate,
            weight_decay=self.cfg.model.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.cfg.training.scheduler.factor,
            patience=self.cfg.training.scheduler.patience, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}