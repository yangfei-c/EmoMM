# 方法公式与策略说明

- 核心符号：q、r、s、z_q、μ_q、z_i、y_i、p_c、τ_q^tag、U(i)、L_K
- 当前情绪估计：z_q = f_text(q_current)
- 目标情绪生成：μ_q = (1-s)z_q + s a_r
- 音乐伪属性预测：z_i = f_VA(m_i)，y_i = f_tag(m_i)
- Tag 原型：基于 y_i,c 对 z_i 加权平均；Top-5% 高置信样本为优选实施方式（见 `figure/data/03_tag_prototype/tag_prototype.csv`）
- 目标 Tag 分布：由 μ_q 与 Tag 原型距离生成
- 联合排序公式：U(i)= -α||z_i-μ_q||² - β||z_i-z_q||² + γ y_i^T τ_q^tag
- 四类策略：maintain、calm、comfort、energize
- 说明：四类策略参考情绪调节理论与音乐情绪调节功能，用于计算建模的目标压缩，不属于临床干预策略

