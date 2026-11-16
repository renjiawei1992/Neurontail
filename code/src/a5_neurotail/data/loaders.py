"""Loaders for FLORES, Belebele, native CSVs, and OPUS-style TSVs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .normalize import normalize_text
from .schema import MCExample, ParallelExample


def _find_flores_split_dir(flores_dir: str | Path, split: str = "dev") -> Path:
    p = Path(flores_dir)
    candidates = [
        p / split,
        p / "flores200_dataset" / split,
        p / "extracted" / "flores200_dataset" / split,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        f"cannot find FLORES split '{split}' under {p}; tried: {[str(c) for c in candidates]}"
    )


def load_flores_texts(
    flores_dir: str | Path, code: str, split: str = "dev", normalize: bool = True
) -> list[str]:
    d = _find_flores_split_dir(flores_dir, split)
    p = d / f"{code}.{split}"
    if not p.exists():
        raise FileNotFoundError(f"FLORES file not found: {p}")
    lines = [line.rstrip("\n") for line in open(p, encoding="utf-8")]
    if normalize:
        lines = [normalize_text(x) for x in lines]
    return lines


def load_flores_parallel(
    flores_dir: str | Path,
    code: str,
    pivot: str = "eng_Latn",
    split: str = "dev",
    normalize: bool = True,
) -> list[ParallelExample]:
    if code == pivot:
        return []
    src = load_flores_texts(flores_dir, pivot, split, normalize)
    tgt = load_flores_texts(flores_dir, code, split, normalize)
    n = min(len(src), len(tgt))
    return [
        ParallelExample(
            id=f"flores-{split}-{code}-{i:05d}",
            src_lang=pivot,
            tgt_lang=code,
            src_text=src[i],
            tgt_text=tgt[i],
            source="FLORES-200",
            split=split,
        )
        for i in range(n)
    ]


def load_belebele(path: str | Path, language: str, normalize: bool = True) -> list[MCExample]:
    out: list[MCExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            passage = r.get("flores_passage", "")
            question = r.get("question", "")
            options = [r.get(f"mc_answer{j}", "") for j in range(1, 5)]
            ans = int(r.get("correct_answer_num", 1)) - 1
            if normalize:
                passage = normalize_text(passage)
                question = normalize_text(question)
                options = [normalize_text(x) for x in options]
            out.append(
                MCExample(
                    id=f"belebele-{language}-{r.get('link', i)}-{r.get('question_number', 1)}",
                    language=language,
                    passage=passage,
                    question=question,
                    options=options,
                    answer_idx=ans,
                    source="Belebele",
                    split="test",
                    meta={"link": r.get("link"), "dialect": r.get("dialect")},
                )
            )
    return out


def _parse_answer(value: str, options: list[str]) -> int:
    v = str(value).strip()
    if v.isdigit():
        n = int(v)
        if 1 <= n <= 4:
            return n - 1
        if 0 <= n <= 3:
            return n
    if v.upper() in {"A", "B", "C", "D"}:
        return "ABCD".index(v.upper())
    # try exact option match
    for i, opt in enumerate(options):
        if v == opt.strip():
            return i
    raise ValueError(f"cannot parse answer {value!r} for options {options!r}")


def load_native_csv(path: str | Path, language: str, normalize: bool = True) -> list[MCExample]:
    """Flexible native CSV loader.

    Accepts columns: item_id/id, passage, question, option_a..d, correct_answer.
    """
    out: list[MCExample] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            passage = r.get("passage", r.get("context", "")) or ""
            question = r.get("question", "") or ""
            options = [r.get(f"option_{c}", r.get(c, "")) or "" for c in "abcd"]
            ans = _parse_answer(r.get("correct_answer", r.get("answer", "1")), options)
            if normalize:
                passage = normalize_text(passage)
                question = normalize_text(question)
                options = [normalize_text(x) for x in options]
            out.append(
                MCExample(
                    id=r.get("item_id", r.get("id", f"{language}-{i:05d}")),
                    language=language,
                    passage=passage,
                    question=question,
                    options=options,
                    answer_idx=ans,
                    source=r.get("source", "native"),
                    split=r.get("split", "test"),
                    meta={k: v for k, v in r.items() if k not in {"passage", "question", "option_a", "option_b", "option_c", "option_d"}},
                )
            )
    return out


def load_parallel_tsv(
    path: str | Path, src_lang: str, tgt_lang: str, source: str = "OPUS", limit: int | None = None
) -> list[ParallelExample]:
    out: list[ParallelExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            out.append(
                ParallelExample(
                    id=f"{source}-{src_lang}-{tgt_lang}-{i:06d}",
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    src_text=normalize_text(parts[0]),
                    tgt_text=normalize_text(parts[1]),
                    source=source,
                    split="train",
                )
            )
            if limit and len(out) >= limit:
                break
    return out


def iter_texts(paths: Iterable[str | Path]) -> Iterable[str]:
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
