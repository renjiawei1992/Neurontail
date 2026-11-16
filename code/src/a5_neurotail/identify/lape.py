"""LAPE: Language Activation Probability Entropy (Tang et al., ACL 2024)."""
from __future__ import annotations

from typing import Any

import numpy as np


def build_prob_matrix(
    stats_by_language: dict[str, dict[str, dict[str, np.ndarray]]],
    key: str,
    languages: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return p[K, N] and the language order."""
    langs = languages or sorted(stats_by_language)
    rows = []
    for lang in langs:
        if lang not in stats_by_language:
            raise KeyError(f"missing stats for language {lang}")
        d = stats_by_language[lang].get(key)
        if d is None:
            raise KeyError(f"missing key {key} for language {lang}")
        rows.append(d["p_pos"].astype(np.float64))
    return np.stack(rows, axis=0), langs


def lape_scores(p: np.ndarray, smoothing: float = 1e-6) -> np.ndarray:
    """Entropy across languages per neuron. Lower = more language-specific."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, 1.0) + smoothing
    p = p / p.sum(axis=0, keepdims=True)
    return -(p * np.log(p)).sum(axis=0)


def select_lape(
    scores: np.ndarray,
    p: np.ndarray,
    target_idx: int,
    top_frac: float = 0.01,
    min_prob: float = 0.05,
) -> np.ndarray:
    n = max(1, int(round(len(scores) * top_frac)))
    eligible = np.where(p[target_idx] >= min_prob)[0]
    if len(eligible) == 0:
        eligible = np.arange(len(scores))
    order = eligible[np.argsort(scores[eligible])]
    return np.sort(order[:n])


def run_lape(
    stats_by_language: dict[str, dict[str, dict[str, np.ndarray]]],
    key: str,
    target_language: str,
    languages: list[str] | None = None,
    top_frac: float = 0.01,
    min_prob: float = 0.05,
    smoothing: float = 1e-6,
) -> dict[str, Any]:
    p, langs = build_prob_matrix(stats_by_language, key, languages)
    scores = lape_scores(p, smoothing)
    idx = int(langs.index(target_language))
    selected = select_lape(scores, p, idx, top_frac, min_prob)
    return {
        "method": "lape",
        "key": key,
        "target_language": target_language,
        "languages": langs,
        "scores": scores.tolist(),
        "selected": selected.tolist(),
        "target_prob": p[idx].tolist(),
    }
