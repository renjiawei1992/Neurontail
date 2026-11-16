"""Torch-free I/O for cached activation statistics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_stats(path: str | Path, stats: dict[str, dict[str, np.ndarray]], meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key, d in stats.items():
        for stat, arr in d.items():
            arrays[f"{key}__{stat}"] = arr
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_stats(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    data = np.load(path, allow_pickle=False)
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in data.files:
        key, stat = name.split("__", 1)
        out.setdefault(key, {})[stat] = data[name]
    return out


def mean_activation(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: d["mean"] for k, d in stats.items()}


def activation_probability(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: d["p_pos"] for k, d in stats.items()}
