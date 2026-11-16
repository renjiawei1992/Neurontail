"""Text and tokenizer fertility statistics."""
from __future__ import annotations

from typing import Any, Iterable

from .normalize import detect_script


def text_stats(texts: Iterable[str]) -> dict[str, Any]:
    texts = list(texts)
    text = "\n".join(texts)
    words = len(text.split())
    chars = len(text)
    by = len(text.encode("utf-8"))
    return {
        "lines": len(texts),
        "words": words,
        "chars": chars,
        "bytes": by,
        "chars_per_word": round(chars / max(words, 1), 4),
        "bytes_per_word": round(by / max(words, 1), 4),
        "bytes_per_char": round(by / max(chars, 1), 4),
        "script": detect_script(text),
    }


def tokenizer_fertility(tokenizer: Any, texts: Iterable[str]) -> dict[str, Any]:
    """Token-level fertility. Requires a HuggingFace tokenizer."""
    texts = list(texts)
    n_tokens = 0
    n_words = 0
    n_bytes = 0
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        n_tokens += len(ids)
        n_words += len(t.split())
        n_bytes += len(t.encode("utf-8"))
    return {
        "lines": len(texts),
        "tokens": n_tokens,
        "words": n_words,
        "bytes": n_bytes,
        "tokens_per_word": round(n_tokens / max(n_words, 1), 4),
        "tokens_per_char": round(n_tokens / max(sum(len(t) for t in texts), 1), 4),
        "bytes_per_token": round(n_bytes / max(n_tokens, 1), 4),
    }


def compute_fertility_table(
    texts_by_language: dict[str, list[str]], tokenizer: Any | None = None
) -> list[dict[str, Any]]:
    rows = []
    for lang, texts in texts_by_language.items():
        row = {"language": lang, **text_stats(texts)}
        if tokenizer is not None:
            row.update({f"tok_{k}": v for k, v in tokenizer_fertility(tokenizer, texts).items() if k != "lines"})
        rows.append(row)
    return rows
