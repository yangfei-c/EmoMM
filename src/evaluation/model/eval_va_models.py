import torch
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error


def pearson_corr(y_true, y_pred):
    """计算皮尔逊相关系数"""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    num = np.sum((y_true - mean_true) * (y_pred - mean_pred))
    den = np.sqrt(np.sum((y_true - mean_true) ** 2) * np.sum((y_pred - mean_pred) ** 2))
    return num / (den + 1e-8)


def compute_ccc(y_true, y_pred):
    """计算一致性相关系数 (CCC)"""
    p = pearson_corr(y_true, y_pred)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    num = 2 * p * std_true * std_pred
    den = std_true ** 2 + std_pred ** 2 + (mean_pred - mean_true) ** 2
    return num / (den + 1e-8)


def get_regression_metrics(pt_path):
    if not os.path.exists(pt_path):
        return None

    data = torch.load(pt_path, weights_only=False)
    preds = data['predictions'].numpy()
    targets = data['targets'].numpy()

    # 拆分 Valence 和 Arousal (列0 和 列1)
    preds_v, preds_a = preds[:, 0], preds[:, 1]
    targets_v, targets_a = targets[:, 0], targets[:, 1]

    metrics = {
        'V_CCC': compute_ccc(targets_v, preds_v),
        'A_CCC': compute_ccc(targets_a, preds_a),
        'V_MSE': mean_squared_error(targets_v, preds_v),
        'A_MSE': mean_squared_error(targets_a, preds_a),
        'V_MAE': mean_absolute_error(targets_v, preds_v),
        'A_MAE': mean_absolute_error(targets_a, preds_a),
    }
    return metrics


if __name__ == "__main__":
    base_dir = r"/outputs/results"

    models = [
        ("Music VA (Audio)", os.path.join(base_dir, "MusicVAPred", "results.pt")),
        ("Text VA (XLM-R)", os.path.join(base_dir, "TextVAPred", "results.pt")),
    ]

    print("\n" + "=" * 90)
    print("📊 情感坐标回归实验结果 (Valence-Arousal Regression Results)".center(80))
    print("=" * 90)

    # 打印表头
    headers = ["Task Module", "Valence CCC(↑)", "Arousal CCC(↑)", "Valence MAE(↓)", "Arousal MAE(↓)"]
    print(f"| {headers[0]:<18} | {headers[1]:<14} | {headers[2]:<14} | {headers[3]:<14} | {headers[4]:<14} |")
    print(f"|:{'-' * 18}-|:{'-' * 14}-|:{'-' * 14}-|:{'-' * 14}-|:{'-' * 14}-|")

    # 提取并打印数据
    for name, path in models:
        m = get_regression_metrics(path)
        if m:
            print(
                f"| **{name:<16}** | {m['V_CCC']:^14.4f} | {m['A_CCC']:^14.4f} | {m['V_MAE']:^14.4f} | {m['A_MAE']:^14.4f} |")
        else:
            print(f"| {name:<18} | {'未生成':^14} | {'-':^14} | {'-':^14} | {'-':^14} |")

    print("=" * 90 + "\n")