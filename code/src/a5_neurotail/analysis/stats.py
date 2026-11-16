"""Mixed-effects, bootstrap, change-point, and multiple-comparison helpers."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def mixed_effects(
    df: pd.DataFrame,
    outcome: str,
    fixed: list[str],
    group: str = "language",
    random_slope: str | None = None,
) -> dict[str, Any]:
    """Fit a MixedLM; returns params, p-values, and a text summary."""
    import statsmodels.formula.api as smf

    data = df.dropna(subset=[outcome] + fixed + [group]).copy()
    formula = f"{outcome} ~ " + " + ".join(fixed)
    vc = f"1 + {random_slope}" if random_slope else "1"
    model = smf.mixedlm(formula, data, groups=data[group], vc_formula=None, re_formula=vc)
    try:
        fit = model.fit(reml=False, method="lbfgs")
    except Exception:
        fit = model.fit(reml=True)
    return {
        "params": fit.params.to_dict(),
        "pvalues": fit.pvalues.to_dict(),
        "summary": str(fit.summary()),
        "n": int(len(data)),
    }


def bootstrap_ci(
    values: Iterable[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    x = np.asarray(list(values), dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def change_point(x: Iterable[float], y: Iterable[float], min_segment: int = 3) -> dict[str, Any]:
    """Single-breakpoint segmented regression by grid search."""
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    best = None
    for i in range(min_segment, len(x) - min_segment):
        left = np.polyfit(x[:i], y[:i], 1)
        right = np.polyfit(x[i:], y[i:], 1)
        sse = ((y[:i] - np.polyval(left, x[:i])) ** 2).sum() + (
            (y[i:] - np.polyval(right, x[i:])) ** 2
        ).sum()
        if best is None or sse < best["sse"]:
            best = {
                "break_index": int(i),
                "break_x": float(x[i]),
                "left_slope": float(left[0]),
                "right_slope": float(right[0]),
                "sse": float(sse),
            }
    return best or {}


def bh_correction(pvalues: Iterable[float], alpha: float = 0.05) -> list[bool]:
    p = np.asarray(list(pvalues), dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    rejected = np.zeros(n, dtype=bool)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    if passed.any():
        k = np.max(np.where(passed)[0])
        rejected[order[: k + 1]] = True
    return rejected.tolist()


def ttest(a: Iterable[float], b: Iterable[float]) -> dict[str, float]:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"t": float(t), "p": float(p), "d": cohens_d(a, b)}
