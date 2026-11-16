"""Prompt templates for language forcing, MCQA, and translation."""
from __future__ import annotations

from typing import Any

from .schema import PromptExample

_SEEDS = [
    "The weather is nice today.",
    "My family lives in a small town near the sea.",
    "She reads a book every evening before dinner.",
    "The train arrives at eight in the morning.",
    "We should protect the environment for future generations.",
    "He works at a hospital in the capital city.",
    "The children are playing football in the park.",
    "This restaurant is famous for its traditional food.",
    "I would like a cup of tea, please.",
    "The museum opens at nine and closes at five.",
]

_TEMPLATES = [
    "Translate the following sentence into {lang}: {seed}",
    "Write the following sentence in {lang}: {seed}",
    "Please express this idea in {lang}: {seed}",
    "Give the {lang} version of this sentence: {seed}",
    "Answer in {lang}. Translate: {seed}",
    "Render the sentence below in {lang}: {seed}",
    "In {lang}, how would you say: {seed}",
    "Provide a natural {lang} translation of: {seed}",
]


def language_forcing_prompts(language_name: str, n: int = 100) -> list[PromptExample]:
    out: list[PromptExample] = []
    for i in range(n):
        seed = _SEEDS[i % len(_SEEDS)]
        tmpl = _TEMPLATES[(i // len(_SEEDS)) % len(_TEMPLATES)]
        out.append(
            PromptExample(
                id=f"lss-{language_name.lower().replace(' ', '_')}-{i:04d}",
                language=language_name,
                prompt=tmpl.format(lang=language_name, seed=seed),
                task="language_forcing",
                meta={"seed": seed, "template": tmpl},
            )
        )
    return out


def mcqa_prompt(passage: str, question: str, options: list[str]) -> str:
    labels = "ABCD"
    lines = [f"Passage: {passage}", f"Question: {question}"]
    for i, opt in enumerate(options):
        lines.append(f"{labels[i]}. {opt}")
    lines.append("Answer:")
    return "\n".join(lines)


def translation_prompt(text: str, target_language: str, source_language: str = "English") -> str:
    return f"Translate the following {source_language} text into {target_language}:\n{text}\nTranslation:"


def instruction_prompts(language_name: str, n: int = 40) -> list[PromptExample]:
    base = [
        "List three benefits of walking every day.",
        "Explain in two sentences why sleep is important.",
        "Write a short greeting for a new colleague.",
        "Summarize the following idea in one sentence: practice improves skill.",
        "Give one example of a healthy breakfast.",
    ]
    out: list[PromptExample] = []
    for i in range(n):
        text = base[i % len(base)]
        out.append(
            PromptExample(
                id=f"instr-{language_name.lower().replace(' ', '_')}-{i:04d}",
                language=language_name,
                prompt=f"Answer in {language_name}. {text}",
                task="instruction",
                meta={"base": text},
            )
        )
    return out


def format_prompt(tokenizer: Any, user_text: str, system: str | None = None) -> str:
    """Use the tokenizer's chat template when available, else plain text."""
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_text})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_text
