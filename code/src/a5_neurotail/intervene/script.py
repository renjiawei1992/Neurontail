"""Script-level control: romanization and script-direction helpers."""
from __future__ import annotations

from typing import Iterable

from ..data.normalize import romanize


def romanize_texts(texts: Iterable[str], backend: str = "icu") -> list[str]:
    return [romanize(t, backend=backend) for t in texts]


def script_pair(texts: list[str]) -> list[tuple[str, str]]:
    """Return (native, romanized) pairs for script-controlled comparisons."""
    roman = romanize_texts(texts)
    return list(zip(texts, roman))
