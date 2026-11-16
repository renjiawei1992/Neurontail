"""Paper-table helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_csg(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = by or ["language", "model"]
    need = {"csg", *by}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return (
        df.groupby(by)["csg"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )


def summarize_lss(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = by or ["language", "model"]
    return df.groupby(by)["lss"].agg(["mean", "std", "count"]).reset_index()


def save_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=index)
    elif path.suffix == ".md":
        path.write_text(df.to_markdown(index=index), encoding="utf-8")
    else:
        df.to_csv(path, index=index)


def to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)
