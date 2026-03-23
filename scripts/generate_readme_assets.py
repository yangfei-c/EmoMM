from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs" / "results"
ASSET_DIR = ROOT / "docs" / "readme_assets"


def _load_pt(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return torch.load(path, weights_only=False)


def _pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    num = np.sum((y_true - mean_true) * (y_pred - mean_pred))
    den = np.sqrt(np.sum((y_true - mean_true) ** 2) * np.sum((y_pred - mean_pred) ** 2))
    return float(num / (den + 1e-8))


def _ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = _pearson_corr(y_true, y_pred)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    num = 2 * p * std_true * std_pred
    den = std_true**2 + std_pred**2 + (mean_pred - mean_true) ** 2
    return float(num / (den + 1e-8))


def collect_music_tag_metrics() -> dict:
    model_files = {
        "MLP_BCE": RESULTS_DIR / "MusicTag" / "MLP_BCE" / "results.pt",
        "MLP_ASL": RESULTS_DIR / "MusicTag" / "MLP_ASL" / "results.pt",
        "MLP_DynamASCL": RESULTS_DIR / "MusicTag" / "MLP_DynamASCL" / "results.pt",
        "Transformer_BCE": RESULTS_DIR / "MusicTag" / "Transformer_BCE" / "results.pt",
        "Transformer_DynamASCL": RESULTS_DIR / "MusicTag" / "Transformer_DynamASCL" / "results.pt",
    }

    task_slices = {"genre": (0, 10), "instrument": (10, 17), "mood": (17, 26)}
    out: dict[str, dict[str, dict[str, float]]] = {}

    for model_name, path in model_files.items():
        data = _load_pt(path)
        model_metrics: dict[str, dict[str, float]] = {}
        for task, (s, e) in task_slices.items():
            y_true = data["targets"][:, s:e].numpy()
            y_prob = data[f"{task}_predictions"].numpy()
            auc = float(roc_auc_score(y_true, y_prob, average="macro"))
            best_f1 = 0.0
            best_thr = 0.5
            for thr in np.linspace(0.01, 0.50, 50):
                y_bin = (y_prob > thr).astype(int)
                f1 = float(f1_score(y_true, y_bin, average="macro", zero_division=0))
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = float(thr)
            model_metrics[task] = {"auc": auc, "best_f1": best_f1, "best_thr": best_thr}
        out[model_name] = model_metrics
    return out


def collect_va_metrics() -> dict:
    model_files = {
        "MusicVA": RESULTS_DIR / "MusicVAPred_m11" / "results.pt",
        "TextVA": RESULTS_DIR / "TextVAPred_m11" / "results.pt",
    }

    out = {}
    for model_name, path in model_files.items():
        data = _load_pt(path)
        pred = data["predictions"].numpy()
        y = data["targets"].numpy()
        out[model_name] = {
            "v_ccc": _ccc(y[:, 0], pred[:, 0]),
            "a_ccc": _ccc(y[:, 1], pred[:, 1]),
            "v_mae": float(mean_absolute_error(y[:, 0], pred[:, 0])),
            "a_mae": float(mean_absolute_error(y[:, 1], pred[:, 1])),
        }
    return out


def plot_music_tag(metrics: dict) -> None:
    models = list(metrics.keys())
    tasks = ["genre", "instrument", "mood"]

    auc = np.array([[metrics[m][t]["auc"] for t in tasks] for m in models])
    f1 = np.array([[metrics[m][t]["best_f1"] for t in tasks] for m in models])

    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)

    for idx, task in enumerate(tasks):
        bars = axes[0].bar(x + (idx - 1) * width, auc[:, idx], width=width, label=task)
        for b in bars:
            y = b.get_height()
            axes[0].text(
                b.get_x() + b.get_width() / 2,
                y + 0.004,
                f"{y:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0].set_title("MusicTag Macro-AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylim(0.65, 0.92)
    axes[0].legend()

    for idx, task in enumerate(tasks):
        bars = axes[1].bar(x + (idx - 1) * width, f1[:, idx], width=width, label=task)
        for b in bars:
            y = b.get_height()
            axes[1].text(
                b.get_x() + b.get_width() / 2,
                y + 0.01,
                f"{y:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[1].set_title("MusicTag Best Macro-F1 (threshold search)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].set_ylim(0.0, 0.62)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "music_tag.png", bbox_inches="tight")
    plt.close(fig)


def plot_va_metrics(metrics: dict) -> None:
    models = list(metrics.keys())
    x = np.arange(len(models))
    width = 0.36

    v_ccc = [metrics[m]["v_ccc"] for m in models]
    a_ccc = [metrics[m]["a_ccc"] for m in models]
    v_mae = [metrics[m]["v_mae"] for m in models]
    a_mae = [metrics[m]["a_mae"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)

    bars_v_ccc = axes[0].bar(x - width / 2, v_ccc, width=width, label="Valence CCC")
    bars_a_ccc = axes[0].bar(x + width / 2, a_ccc, width=width, label="Arousal CCC")
    axes[0].set_title("VA Regression ([-1,1]): CCC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylim(0.60, 0.90)
    axes[0].legend()
    for b in list(bars_v_ccc) + list(bars_a_ccc):
        y = b.get_height()
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            y + 0.006,
            f"{y:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    bars_v_mae = axes[1].bar(x - width / 2, v_mae, width=width, label="Valence MAE")
    bars_a_mae = axes[1].bar(x + width / 2, a_mae, width=width, label="Arousal MAE")
    axes[1].set_title("VA Regression ([-1,1]): MAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].set_ylim(0.0, 0.22)
    axes[1].legend()
    for b in list(bars_v_mae) + list(bars_a_mae):
        y = b.get_height()
        axes[1].text(
            b.get_x() + b.get_width() / 2,
            y + 0.004,
            f"{y:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "music_text_va.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    music_tag = collect_music_tag_metrics()
    va = collect_va_metrics()

    plot_music_tag(music_tag)
    plot_va_metrics(va)

    print(f"Generated assets in: {ASSET_DIR}")


if __name__ == "__main__":
    main()

