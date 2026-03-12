import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings("ignore")

def get_metrics(pt_path):
    data = torch.load(pt_path, weights_only=False)
    results = {}

    for task in ['genre', 'instrument', 'mood']:
        if task == 'genre':
            targets = data['targets'][:, :10]
        elif task == 'instrument':
            targets = data['targets'][:, 10:17]
        else:
            targets = data['targets'][:, 17:]

        pred_key = f"{task}_predictions"
        preds_prob = data[pred_key].numpy()
        targets_np = targets.numpy()
        try:
            auc = roc_auc_score(targets_np, preds_prob, average='macro')
            best_f1 = 0
            for thresh in np.linspace(0.01, 0.5, 50):
                current_preds_bin = (preds_prob > thresh).astype(int)
                current_f1 = f1_score(targets_np, current_preds_bin, average='macro', zero_division=0)
                if current_f1 > best_f1: best_f1 = current_f1
            results[task] = (auc, best_f1)
        except:
            results[task] = (0.0, 0.0)

    return results

if __name__ == "__main__":
    base_dir = r"D:\AAAAAAAAAA_emotion\Code\EmoMusicRec\outputs\results"

    models = [
        ("MLP + BCE", r"MusicTag\MLP_BCE\results.pt"),
        ("MLP + ASL", r"MusicTag\MLP_ASL\results.pt"),
        ("MLP + DynamASCL", r"MusicTag\MLP_DynamASCL\results.pt"),
        ("Transformer + BCE", r"MusicTag\Transformer_BCE\results.pt"),
        ("Transformer + DynamASCL", r"MusicTag\Transformer_DynamASCL\results.pt"),
    ]

    all_results = []
    for name, rel_path in models:
        full_path = os.path.join(base_dir, rel_path)
        metrics = get_metrics(full_path)
        if metrics:
            all_results.append({
                'name': name,
                'g_auc': metrics['genre'][0],
                'g_f1': metrics['genre'][1],
                'i_auc': metrics['instrument'][0],
                'i_f1': metrics['instrument'][1],
                'm_auc': metrics['mood'][0],
                'm_f1': metrics['mood'][1],
            })
        else:
            all_results.append({'name': name, 'missing': True})

    max_scores = {
        'g_auc': max([r.get('g_auc', 0) for r in all_results]),
        'g_f1': max([r.get('g_f1', 0) for r in all_results]),
        'i_auc': max([r.get('i_auc', 0) for r in all_results]),
        'i_f1': max([r.get('i_f1', 0) for r in all_results]),
        'm_auc': max([r.get('m_auc', 0) for r in all_results]),
        'm_f1': max([r.get('m_f1', 0) for r in all_results]),
    }

    print("\n" + "=" * 105)
    print("📊 实验结果 ( Study Results)".center(95))
    print("=" * 105)

    # 设定列宽
    col_widths = [28, 12, 12, 12, 12, 12, 12]
    headers = ["Model Configuration", "Genre AUC", "Genre F1", "Inst. AUC", "Inst. F1", "Mood AUC", "Mood F1"]

    # 打印表头
    header_str = "| " + " | ".join([f"{h:^{w}}" for h, w in zip(headers, col_widths)]) + " |"
    sep_str = f"|:{'-' * col_widths[0]}|" + "|".join([f":{'-' * w}:" for w in col_widths[1:]]) + "|"

    print(header_str)
    print(sep_str)

    # 打印数据行
    for r in all_results:
        if r.get('missing'):
            print(f"| {r['name']:<28} | {'未生成':^12} | {'-':^12} | {'-':^12} | {'-':^12} | {'-':^12} | {'-':^12} |")
            continue

        def fmt(val, key, width):
            s = f"{val:.4f}"
            if val == max_scores[key] and val > 0:
                s = f"**{s}**"
            return f"{s:^{width}}"

        row_str = (
            f"| **{r['name']:<25}** "
            f"| {fmt(r['g_auc'], 'g_auc', col_widths[1])} "
            f"| {fmt(r['g_f1'], 'g_f1', col_widths[2])} "
            f"| {fmt(r['i_auc'], 'i_auc', col_widths[3])} "
            f"| {fmt(r['i_f1'], 'i_f1', col_widths[4])} "
            f"| {fmt(r['m_auc'], 'm_auc', col_widths[5])} "
            f"| {fmt(r['m_f1'], 'm_f1', col_widths[6])} |"
        )
        print(row_str)

    print("=" * 105 + "\n")