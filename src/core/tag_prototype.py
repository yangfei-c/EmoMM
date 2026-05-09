from __future__ import annotations

import numpy as np


def compute_tag_prototypes(tracks, top_percent: float = 0.05):
    z_matrix = np.asarray([t["va"] for t in tracks], dtype=np.float32)
    y_matrix = np.asarray([t["tag26"] for t in tracks], dtype=np.float32)

    num_tags = y_matrix.shape[1]
    prototypes = []

    for j in range(num_tags):
        probs = y_matrix[:, j]
        threshold = np.quantile(probs, 1.0 - top_percent)
        idx = probs >= threshold

        weights = probs[idx]
        z_selected = z_matrix[idx]

        if weights.sum() <= 1e-8:
            proto = z_matrix.mean(axis=0)
        else:
            proto = (z_selected * weights[:, None]).sum(axis=0) / (weights.sum() + 1e-8)

        prototypes.append(proto)
    return np.asarray(prototypes, dtype=np.float32)