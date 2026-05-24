# 训练配置简表

- Text VA 模型
  - 数据集：见 `figure/data/01_training_models/train.md`
  - 模型结构：见 `figure/data/01_training_models/train.md`
  - 输入：文本（说明见 `figure/data/01_training_models/train.md`）
  - 输出：2 维 VA 预测（模型预测属性）
  - 损失函数：见 `figure/data/01_training_models/train.md`
  - 结果文件：`figure/data/01_training_models/train.md`，`figure/data/01_training_models/training_metrics.csv`

- Music VA 模型
  - 数据集：见 `figure/data/01_training_models/train.md`
  - 模型结构：见 `figure/data/01_training_models/train.md`
  - 输入：音乐音频特征（说明见 `figure/data/01_training_models/train.md`）
  - 输出：2 维 VA 预测（模型预测属性）
  - MERT 是否冻结：未在 `figure/data/01_training_models/train.md` 中明确，需以训练配置为准
  - 损失函数：见 `figure/data/01_training_models/train.md`
  - 结果文件：`figure/data/01_training_models/train.md`，`figure/data/01_training_models/training_metrics.csv`

- Music Tag 模型
  - 数据集：见 `figure/data/01_training_models/train.md`
  - 标签数量：26（标签列表见 `figure/data/02_semantic_library/semantic.md`）
  - 模型结构：见 `figure/data/01_training_models/train.md`
  - 输入：音乐音频特征（说明见 `figure/data/01_training_models/train.md`）
  - 输出：26 维 Tag 预测概率（模型预测伪属性）
  - MERT 是否冻结：未在 `figure/data/01_training_models/train.md` 中明确，需以训练配置为准
  - 损失函数：见 `figure/data/01_training_models/train.md`
  - 最终选择：见 `figure/data/01_training_models/train.md`
  - 注意事项：Tag 为模型预测伪属性，不是人工标注

