"""Template-count anomaly spikes: robust z-score of windowed counts (median/MAD)."""
from __future__ import annotations

import numpy as np


def window_counts(cids: list[int], n_clusters: int, windows: int = 40) -> np.ndarray:
    m = np.zeros((n_clusters, windows), dtype=int)
    n = len(cids)
    for i, c in enumerate(cids):
        m[c, min(windows - 1, i * windows // n)] += 1
    return m


def spikes(series: np.ndarray, z: float = 4.0, min_count: int = 5) -> list[int]:
    med = np.median(series)
    mad = np.median(np.abs(series - med)) or 1.0
    score = 0.6745 * (series - med) / mad
    return [int(i) for i in np.where((score > z) & (series >= min_count))[0]]
