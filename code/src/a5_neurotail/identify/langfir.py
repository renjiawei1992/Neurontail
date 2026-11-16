"""LangFIR-style random-token control for language-specific units."""
from __future__ import annotations

from typing import Any

import numpy as np


def random_token_texts(tokenizer, n: int = 512, length: int = 64, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    vocab_size = len(tokenizer)
    special = set(tokenizer.all_special_ids or [])
    ids = [i for i in range(vocab_size) if i not in special]
    out = []
    for _ in range(n):
        sample = rng.choice(ids, size=length, replace=True).tolist()
        out.append(tokenizer.decode(sample, skip_special_tokens=True))
    return out


def langfir_scores(
    lang_stats: dict[str, dict[str, np.ndarray]],
    rand_stats: dict[str, dict[str, np.ndarray]],
    key: str,
    margin: float = 0.05,
) -> np.ndarray:
    """p_lang - p_random; higher = more language-specific relative to random tokens."""
    p_lang = lang_stats[key]["p_pos"].astype(np.float64)
    p_rand = rand_stats[key]["p_pos"].astype(np.float64)
    return p_lang - p_rand - margin


def select_langfir(
    scores: np.ndarray,
    p_lang: np.ndarray,
    p_rand: np.ndarray,
    top_frac: float = 0.01,
    min_prob: float = 0.05,
    random_pos_threshold: float = 0.5,
) -> np.ndarray:
    n = max(1, int(round(len(scores) * top_frac)))
    eligible = np.where((p_lang >= min_prob) & (p_rand <= random_pos_threshold) & (scores > 0))[0]
    if len(eligible) == 0:
        eligible = np.where(scores > 0)[0]
    if len(eligible) == 0:
        return np.array([], dtype=int)
    order = eligible[np.argsort(scores[eligible])[::-1]]
    return np.sort(order[:n])


def run_langfir(
    lang_stats: dict[str, dict[str, np.ndarray]],
    rand_stats: dict[str, dict[str, np.ndarray]],
    key: str,
    top_frac: float = 0.01,
    min_prob: float = 0.05,
    random_pos_threshold: float = 0.5,
    margin: float = 0.05,
) -> dict[str, Any]:
    scores = langfir_scores(lang_stats, rand_stats, key, margin)
    p_lang = lang_stats[key]["p_pos"]
    p_rand = rand_stats[key]["p_pos"]
    selected = select_langfir(scores, p_lang, p_rand, top_frac, min_prob, random_pos_threshold)
    return {
        "method": "langfir",
        "key": key,
        "scores": scores.tolist(),
        "selected": selected.tolist(),
        "p_lang": p_lang.tolist(),
        "p_random": p_rand.tolist(),
    }
