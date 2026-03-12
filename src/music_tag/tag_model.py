import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)#生成索引并增加一个维度（max_len,1)

        # 《Attention Is All You Need》论文中公式的分母，1/1000^(2i/d_model)
        two_i=torch.arange(0, d_model, 2)#2i
        div_term = 10000.0 ** (-two_i / d_model)

        pe = torch.zeros(max_len, 1, d_model)#[50,1,256]
        pe[:, 0, 0::2] = torch.sin(position * div_term)#偶数列填正弦
        pe[:, 0, 1::2] = torch.cos(position * div_term)#奇数列填余弦
        self.register_buffer('pe', pe)#存下来但是不是参数不需要更新

    def forward(self, x):#x [batch,len,d]-->[64,12,256]
        seq_len = x.size(1)
        pe_slice=self.pe[:seq_len]#[50,1,768]--->[12,1,256]
        x = x + pe_slice.transpose(0, 1)#[64,12,256]+[1,12,256](transpose,转换第0维和第1维
        return x


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        #用于判断输入特征重要性,256-->128-->1
        self.attention = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask=None):
        '''
        假设进来x,[1,12,256],mask[false,false,false,0,0,……],即进来一首音乐，只有3个片段
        '''
        #[1,12,256]-->[1,12,1]--->[1,12]
        attn_weights = self.attention(x).squeeze(-1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask, -1e9)
            #每首音乐片段数量不一样，配置生成最大12片段。如果一首音乐仅有4个片段，那么通过这个把后面8个值为零训练时不去看
        attn_weights = torch.softmax(attn_weights, dim=1)
        #[1,1,12]与[1,12,256]进行批次矩阵乘法，[1,12]×[12,256]
        pooled_out = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        #浓缩后的[64,256]
        return pooled_out, attn_weights


class TagModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_type = cfg.model_type

        input_dim = cfg.model.input_dim
        dropout = cfg.model.dropout
        latent_dim = cfg.model.latent_dim

        # ================= [ 注意力机制 ] =================
        if self.model_type == "Transformer":
            d_model = 256
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model=d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=1024, dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)#片段之间互相焦虑
            self.attention_pool = AttentionPooling(d_model=d_model)
            proj_in_dim = d_model
        elif self.model_type=="MLP":
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            proj_in_dim = 256
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        # ================= [ 共享：多任务分类头 ] =================
        self.genre_proj = nn.Sequential(nn.Linear(proj_in_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout))
        self.instrument_proj = nn.Sequential(nn.Linear(proj_in_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout))
        self.mood_proj = nn.Sequential(nn.Linear(proj_in_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout))

        self.genre_head = nn.Linear(latent_dim, cfg.task.num_genres)
        self.instrument_head = nn.Linear(latent_dim, cfg.task.num_instruments)
        self.mood_head = nn.Linear(latent_dim, cfg.task.num_moods)

    def forward(self, x, mask=None):
        # 自动分流计算
        if self.model_type == "Transformer":
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            encoded_seq = self.transformer(x, src_key_padding_mask=mask)
            shared_features, attn_weights = self.attention_pool(encoded_seq, mask=mask)
        elif self.model_type=="MLP":
            shared_features = self.backbone(x)
            attn_weights = None
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        z_genre = self.genre_proj(shared_features)
        z_instrument = self.instrument_proj(shared_features)
        z_mood = self.mood_proj(shared_features)

        return {
            'genre': self.genre_head(z_genre),
            'instrument': self.instrument_head(z_instrument),
            'mood': self.mood_head(z_mood),
            'features': torch.cat([z_genre, z_instrument, z_mood], dim=1),
            'attn_weights': attn_weights
        }




import hydra
@hydra.main(version_base=None, config_path="../../configs", config_name="music_tag")
def main(cfg):
    model=TagModel(cfg)
    print(model)
if __name__=="__main__":
    main()
