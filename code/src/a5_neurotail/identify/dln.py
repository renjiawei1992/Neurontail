"""Distribution-aware Language Neuron identification (DLN).

Approximates pairwise activation-distribution overlap from histograms.
If histograms are unavailable, falls back to overlap of activation probabilities.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _norm_hist(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    s = h.sum(axis=-1, keepdims=True)
    return h / np.maximum(s, 1e-12)


def overlap_coefficient(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Per-neuron histogram intersection; h1/h2 are [N, bins]."""
    a = _norm_hist(h1)
    b = _norm_hist(h2)
    return np.minimum(a, b).sum(axis=-1)


def dln_scores(
    stats_by_language: dict[str, dict[str, dict[str, np.ndarray]]],
    key: str,
    languages: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return specificity[K, N]: 1 - max overlap with any other language."""
    langs = languages or sorted(stats_by_language)
    if not all("hist" in stats_by_language[l][key] for l in langs):
        # fallback: probability-vector overlap
        p = np.stack([stats_by_language[l][key]["p_pos"] for l in langs], axis=0)
        p = np.clip(p, 0, 1)
        spec = np.zeros_like(p)
        for i in range(len(langs)):
            others = np.delete(p, i, axis=0)
            # overlap of Bernoulli probabilities, bounded in [0,1]
            ov = np.minimum(p[i][None, :], others).sum(axis=0) / np.maximum(
                p[i][None, :] + others, 1e-12
            ).sum(axis=0)
            spec[i] = 1.0 - ov
        return spec, langs
    hists = [stats_by_language[l][key]["hist"] for l in langs]
    spec = np.zeros((len(langs), hists[0].shape[0]), dtype=np.float64)
    for i in range(len(langs)):
        ovs = [overlap_coefficient(hists[i], hists[j]) for j in range(len(langs)) if j != i]
        spec[i] = 1.0 - np.max(np.stack(ovs, axis=0), axis=0)
    return spec, langs


def select_dln(spec: np.ndarray, target_idx: int, top_frac: float = 0.01) -> np.ndarray:
    n = max(1, int(round(spec.shape[1] * top_frac)))
    order = np.argsort(spec[target_idx])[::-1]
    return np.sort(order[:n])


def run_dln(
    stats_by_language: dict[str, dict[str, dict[str, np.ndarray]]],
    key: str,
    target_language: str,
    languages: list[str] | None = None,
    top_frac: float = 0.01,
) -> dict[str, Any]:
    spec, langs = dln_scores(stats_by_language, key, languages)
    idx = int(langs.index(target_language))
    selected = select_dln(spec, idx, top_frac)
    return {
        "method": "dln",
        "key": key,
        "target_language": target_language,
        "languages": langs,
        "scores": spec.tolist(),
        "selected": selected.tolist(),
    }
