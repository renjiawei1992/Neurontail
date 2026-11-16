"""Donor transfer gains and matrices."""
from __future__ import annotations

import numpy as np
import pandas as pd


def transfer_gain(base_score: float, patched_score: float) -> float:
    return float(patched_score - base_score)


def build_transfer_matrix(records: list[dict]) -> pd.DataFrame:
    """records: target, donor_type, donor_lang, gain."""
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.pivot_table(
        index="target",
        columns=["donor_type", "donor_lang"],
        values="gain",
        aggfunc="mean",
    )


def script_vs_family_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean gain by script_overlap and family_overlap."""
    need = {"script_overlap", "family_overlap", "gain"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df.groupby(["script_overlap", "family_overlap"])["gain"].agg(["mean", "std", "count"]).reset_index()
