from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import csv
import io

try:
    import torch
except ImportError as exc:  # pragma: no cover - used in runtime only
    raise RuntimeError("torch is required to load results.pt") from exc

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "outputs" / "results"
OUTPUT_MD = ROOT / "figure" / "data" / "01_training_models" / "train.md"
OUTPUT_CSV = ROOT / "figure" / "data" / "01_training_models" / "training_metrics.csv"

VA_RUNS_M11 = [
    "TextVAPred_m11",
    "MusicVAPred_m11",
]

F1_THRESHOLD = 0.5


def torch_load(path: Path) -> Any:
    with warnings_suppressed():
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            return torch.load(path, map_location="cpu", weights_only=False)


class warnings_suppressed:
    def __enter__(self):
        import warnings

        warnings.simplefilter("ignore")

    def __exit__(self, exc_type, exc, tb):
        return False


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=float)


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.linalg.norm(xm, 2) * np.linalg.norm(ym, 2)
    if denom == 0:
        return float("nan")
    return float(np.dot(xm, ym) / (denom + 1e-8))


def ccc_np(x: np.ndarray, y: np.ndarray) -> float:
    corr = pearsonr_np(x, y)
    if math.isnan(corr):
        return float("nan")
    var_x = np.var(x)
    var_y = np.var(y)
    mean_gap2 = float((np.mean(x) - np.mean(y)) ** 2)
    denom = var_x + var_y + mean_gap2
    if denom == 0:
        return float("nan")
    return float((2.0 * corr * np.std(x) * np.std(y)) / (denom + 1e-8))


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def binary_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    y = targets.astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores.astype(float))
    sum_pos = float(ranks[y == 1].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def multilabel_macro_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    aucs = [binary_auc(scores[:, idx], targets[:, idx]) for idx in range(scores.shape[1])]
    valid = [auc for auc in aucs if not math.isnan(auc)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def fmt(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def load_va_metrics(run_name: str) -> dict[str, float]:
    path = RESULTS_DIR / run_name / "results.pt"
    obj = torch_load(path)
    preds = tensor_to_numpy(obj["predictions"])
    targets = tensor_to_numpy(obj["targets"])
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    ccc_v = ccc_np(preds[:, 0], targets[:, 0])
    ccc_a = ccc_np(preds[:, 1], targets[:, 1])
    ccc_mean = float(np.nanmean([ccc_v, ccc_a]))
    return {
        "mse": mse,
        "mae": mae,
        "ccc": ccc_mean,
        "ccc_v": ccc_v,
        "ccc_a": ccc_a,
    }


def parse_tag_run_name(run_name: str) -> tuple[str, str]:
    if "_" in run_name:
        model, loss = run_name.split("_", 1)
        return model, loss
    return run_name, "-"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def macro_f1(scores: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    probs = sigmoid(scores)
    preds = (probs >= threshold).astype(int)
    y_true = targets.astype(int)
    f1s: list[float] = []
    for idx in range(y_true.shape[1]):
        y = y_true[:, idx]
        y_hat = preds[:, idx]
        tp = int(((y == 1) & (y_hat == 1)).sum())
        fp = int(((y == 0) & (y_hat == 1)).sum())
        fn = int(((y == 1) & (y_hat == 0)).sum())
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
            continue
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    if not f1s:
        return float("nan")
    return float(np.mean(f1s))


def load_tag_metrics(run_name: str) -> dict[str, float | None]:
    path = RESULTS_DIR / "MusicTag" / run_name / "results.pt"
    obj = torch_load(path)
    targets = tensor_to_numpy(obj["targets"])
    metrics: dict[str, float | None] = {
        "genre_auc": None,
        "instrument_auc": None,
        "mood_auc": None,
        "genre_f1": None,
        "instrument_f1": None,
        "mood_f1": None,
    }
    offset = 0
    for key in ("genre", "instrument", "mood"):
        pred_key = f"{key}_predictions"
        cat_pred = obj.get(pred_key)
        if cat_pred is None:
            continue
        scores = tensor_to_numpy(cat_pred)
        width = int(scores.shape[1])
        cat_targets = targets[:, offset : offset + width]
        offset += width
        auc_value = multilabel_macro_auc(scores, cat_targets)
        f1_value = macro_f1(scores, cat_targets, F1_THRESHOLD)
        metrics[f"{key}_auc"] = auc_value
        metrics[f"{key}_f1"] = f1_value
    return metrics


def append_va_table(lines: list[str], title: str, runs: list[str]) -> None:
    lines.append(title)
    lines.append("")
    lines.append("| 模块 | MSE | MAE | Valence CCC | Arousal CCC | CCC 均值 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for run_name in runs:
        path = RESULTS_DIR / run_name / "results.pt"
        if not path.exists():
            continue
        m = load_va_metrics(run_name)
        lines.append(
            "| {model} | {mse} | {mae} | {ccc_v} | {ccc_a} | {ccc} |".format(
                model=run_name,
                mse=fmt(m["mse"]),
                mae=fmt(m["mae"]),
                ccc_v=fmt(m["ccc_v"]),
                ccc_a=fmt(m["ccc_a"]),
                ccc=fmt(m["ccc"]),
            )
        )
    lines.append("")



def build_md() -> str:
    lines: list[str] = []


    append_va_table(lines, "表1：VA 测试结果（m11）", VA_RUNS_M11)

    lines.append("表2：MusicTag 测试结果")
    lines.append("")
    lines.append("| 配置 | Genre AUC | Genre F1 | Inst. AUC | Inst. F1 | Mood AUC | Mood F1 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    tag_root = RESULTS_DIR / "MusicTag"
    run_names = sorted([p.name for p in tag_root.iterdir() if (p / "results.pt").exists()])
    for run_name in run_names:
        model, loss = parse_tag_run_name(run_name)
        m = load_tag_metrics(run_name)
        lines.append(
            "| {config} | {genre_auc} | {genre_f1} | {inst_auc} | {inst_f1} | {mood_auc} | {mood_f1} |".format(
                config=f"{model}+{loss}" if loss != "-" else model,
                genre_auc=fmt(m["genre_auc"]),
                genre_f1=fmt(m["genre_f1"]),
                inst_auc=fmt(m["instrument_auc"]),
                inst_f1=fmt(m["instrument_f1"]),
                mood_auc=fmt(m["mood_auc"]),
                mood_f1=fmt(m["mood_f1"]),
            )
        )
    lines.append("")

    return "\n".join(lines)


def build_csv() -> str:
    lines: list[list[str]] = []

    lines.append(["表1：VA 测试结果（m11）"])
    lines.append(["模块", "MSE", "MAE", "Valence CCC", "Arousal CCC", "CCC 均值"])
    for run_name in VA_RUNS_M11:
        path = RESULTS_DIR / run_name / "results.pt"
        if not path.exists():
            continue
        m = load_va_metrics(run_name)
        lines.append(
            [
                run_name,
                fmt(m["mse"]),
                fmt(m["mae"]),
                fmt(m["ccc_v"]),
                fmt(m["ccc_a"]),
                fmt(m["ccc"]),
            ]
        )
    lines.append([])

    lines.append(["表2：MusicTag 测试结果"])
    lines.append(["配置", "Genre AUC", "Genre F1", "Inst. AUC", "Inst. F1", "Mood AUC", "Mood F1"])
    tag_root = RESULTS_DIR / "MusicTag"
    run_names = sorted([p.name for p in tag_root.iterdir() if (p / "results.pt").exists()])
    for run_name in run_names:
        model, loss = parse_tag_run_name(run_name)
        m = load_tag_metrics(run_name)
        lines.append(
            [
                f"{model}+{loss}" if loss != "-" else model,
                fmt(m["genre_auc"]),
                fmt(m["genre_f1"]),
                fmt(m["instrument_auc"]),
                fmt(m["instrument_f1"]),
                fmt(m["mood_auc"]),
                fmt(m["mood_f1"]),
            ]
        )

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerows(lines)
    return buf.getvalue()


def main() -> None:
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(build_md(), encoding="utf-8")
    OUTPUT_CSV.write_text(build_csv(), encoding="utf-8")


if __name__ == "__main__":
    main()
