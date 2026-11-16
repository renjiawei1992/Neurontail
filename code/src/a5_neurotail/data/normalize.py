"""Text normalization and script detection (stdlib only; ICU optional)."""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")

_SCRIPT_RANGES = [
    ("Latin", [(0x0041, 0x024F), (0x1E00, 0x1EFF)]),
    ("Cyrillic", [(0x0400, 0x04FF), (0x0500, 0x052F)]),
    ("Greek", [(0x0370, 0x03FF)]),
    ("Arabic", [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)]),
    ("Hebrew", [(0x0590, 0x05FF)]),
    ("Devanagari", [(0x0900, 0x097F)]),
    ("Bengali", [(0x0980, 0x09FF)]),
    ("Thai", [(0x0E00, 0x0E7F)]),
    ("Tibetan", [(0x0F00, 0x0FFF)]),
    ("Ethiopic", [(0x1200, 0x137F), (0x1380, 0x139F)]),
    ("Thaana", [(0x0780, 0x07BF)]),
    ("Georgian", [(0x10A0, 0x10FF), (0x1C90, 0x1CBF)]),
    ("Hangul", [(0xAC00, 0xD7AF), (0x1100, 0x11FF)]),
    ("Hiragana", [(0x3040, 0x309F)]),
    ("Katakana", [(0x30A0, 0x30FF)]),
    ("Han", [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)]),
]


def normalize_text(text: str, nfc: bool = True, collapse_ws: bool = True) -> str:
    if nfc:
        text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ")
    if collapse_ws:
        text = _WS.sub(" ", text).strip()
    return text


def strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def detect_script(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return "Unknown"
    return max(counts, key=counts.get)


def romanize(text: str, backend: str = "icu") -> str:
    """Romanize using ICU if available; otherwise raise a clear error."""
    if backend != "icu":
        raise ValueError(f"unsupported romanizer: {backend}")
    try:
        import icu  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Romanization requires PyICU. Install with `pip install PyICU` or "
            "provide pre-romanized text."
        ) from exc
    tr = icu.Transliterator.createInstance("Any-Latin; Latin-ASCII")
    return str(tr.transliterate(text))
