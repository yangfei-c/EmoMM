import torch.nn as nn

class MusicVAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.model.input_dim
        # 轻量级特征回归网络
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),  # 强 Dropout 防止在小数据集上过拟合

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2)  # 输出两维：Valence 和 Arousal
        )

        # 边界约束，保证预测的情绪坐标永远在 [0, 1] 之间
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        logits = self.mlp(x)
        return self.sigmoid(logits)