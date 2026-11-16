"""Unified data records and JSONL helpers."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ParallelExample:
    id: str
    src_lang: str
    tgt_lang: str
    src_text: str
    tgt_text: str
    source: str = ""
    split: str = "dev"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCExample:
    id: str
    language: str
    passage: str
    question: str
    options: list[str]
    answer_idx: int
    source: str = ""
    split: str = "test"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptExample:
    id: str
    language: str
    prompt: str
    task: str = "language_forcing"
    meta: dict[str, Any] = field(default_factory=dict)


def to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            d = row if isinstance(row, dict) else asdict(row)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def validate_mc(rows: list[dict[str, Any]]) -> None:
    for i, r in enumerate(rows):
        if len(r.get("options", [])) != 4:
            raise ValueError(f"row {i}: expected 4 options, got {len(r.get('options', []))}")
        if not (0 <= int(r.get("answer_idx", -1)) < 4):
            raise ValueError(f"row {i}: answer_idx out of range")


def validate_parallel(rows: list[dict[str, Any]]) -> None:
    for i, r in enumerate(rows):
        if not r.get("src_text") or not r.get("tgt_text"):
            raise ValueError(f"row {i}: empty src/tgt text")
