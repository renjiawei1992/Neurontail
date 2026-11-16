"""Language Steering Shift (LSS) with a pluggable language identifier."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_ISO_MAP = {
    "eng": "en", "deu": "de", "fra": "fr", "spa": "es", "jpn": "ja", "zho": "zh",
    "kor": "ko", "rus": "ru", "hin": "hi", "tur": "tr", "vie": "vi", "ind": "id",
    "swh": "sw", "tha": "th", "heb": "he", "ukr": "uk", "tsn": "tn", "zul": "zu",
    "amh": "am", "mlt": "mt", "cym": "cy", "isl": "is", "kat": "ka", "bod": "bo",
    "fij": "fj", "mri": "mi", "ton": "to", "bis": "bi", "div": "dv", "niu": "niu",
    "rar": "rar", "pis": "pis",
}


def canonical_lang(code: str) -> str:
    base = code.split("_")[0]
    return _ISO_MAP.get(base, base)


# Lexicon fallback for languages missing from fastText lid.176 (e.g. Fijian).
# Distinctive function/content words; single-letter forms excluded except a few
# very characteristic particles.
_FIJIAN_WORDS = {
    "na", "sa", "se", "ni", "vei", "kei", "me", "ko", "mai", "tiko", "sara",
    "tale", "ga", "vaka", "kana", "vale", "vanua", "wai", "tamata", "gone",
    "levu", "lailai", "dina", "cakacaka", "vola", "vosa", "loma", "matua",
    "yalewa", "tagane", "dua", "rua", "tolu", "va", "lima", "bula", "vinaka",
    "yaga", "cava", "oqo", "oya", "koya", "keimami", "keitou", "kedra",
    "nodra", "era", "au", "iko", "daru", "keda", "marama", "turaga", "tamana",
    "tina", "tabu", "kalou", "siga", "bogi", "mataka", "yakavi", "dabe",
    "moce", "lako", "tiko", "rawa", "vakaviti", "kerekere", "yadra",
    "katakata", "batabata", "draki", "sigani", "matavuvale",
}
_ZULU_WORDS = {
    "umuntu", "abantu", "isizulu", "ukudla", "amanzi", "umfana", "intombazane",
    "kakhulu", "lapha", "yini", "njani", "kahle", "hamba", "bona", "mina",
    "wena", "yena", "thina", "nina", "ngiya", "uxolo", "ngiyabonga", "ekhaya",
    "isikole", "umsebenzi", "uthisha", "umama", "ubaba", "ingane", "izingane",
    "umuntu", "isonto", "ilanga", "inyanga", "umhlaba", "izulu", "umlilo",
    "ukuhamba", "ukubhala", "ukufunda", "kudala", "namuhla", "kusasa", "izolo",
    "ngoba", "kodwa", "futhi", "kanye", "noma", "uma", "ngakho", "phansi",
    "phezulu", "ngaphandle", "ngaphakathi", "kancane", "kuningi", "konke",
    "okunye", "umoya", "inhliziyo", "isandla", "unyawo", "amehlo", "indlebe",
    "umlomo", "ikhanda", "umzimba", "impilo", "ukufa", "umndeni", "umngani",
    "isitha", "inkosi", "indoda", "umfazi", "insizwa", "intombi",
    "uphuthu", "isibongo", "igama", "izwi", "incwadi", "iphepha", "ipeni",
    "umgwaqo", "imoto", "isitimela", "ibhasi", "indiza", "umkhumbi",
}
_MAORI_WORDS = {
    "te", "ngā", "he", "ki", "kei", "me", "ko", "mō", "mā", "whakaaro",
    "whare", "wai", "whenua", "tangata", "tamariki", "kai", "mahi", "kōrero",
    "aroha", "haere", "mai", "nui", "iti", "pai", "tino", "anō", "ka",
    "tēnei", "tērā", "ēnei", "ērā", "iwi", "whānau", "tama", "kōtiro",
    "tāne", "wahine", "rangatira", "tohunga", "haka", "waiata", "karakia",
    "marae", "awa", "maunga", "moana", "rangi", "papa", "whenua", "ora",
    "mate", "roto", "waho", "runga", "raro", "mua", "muri", "tata", "tawhiti",
    "rahi", "iti", "hou", "tawhito", "pai", "kino", "ātaahua", "tere",
    "hīkoi", "kanikani", "kaukau", "moemoeā", "whakapapa", "mana", "tapu",
    "noa", "koha", "hui", "pōwhiri", "haere", "mai", "atu", "ake", "iho",
    "ahau", "koe", "ia", "mātou", "koutou", "rātou", "tātou", "kāore",
}
_LEXICONS = {"fj": _FIJIAN_WORDS, "zu": _ZULU_WORDS, "mi": _MAORI_WORDS}


def lexicon_lid_predict(text: str, target: str) -> LIDPrediction:
    """Fraction of alphabetic tokens matching a language lexicon.

    Includes a degeneracy guard: repetitive token-loop outputs (steering
    collapse) must not pass as real language.
    """
    import re as _re

    words = [w.lower() for w in _re.findall(r"[A-Za-zĀāĒēĪīŌōŪū']+", text)]
    lexicon = _LEXICONS.get(target)
    if not lexicon or len(words) < 3:
        return LIDPrediction("", 0.0)
    # degeneracy guards
    n_unique = len(set(words))
    if n_unique < max(3, 0.3 * len(words)):
        return LIDPrediction("degenerate", 0.0)
    longest_run = 1
    run = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        longest_run = max(longest_run, run)
    if longest_run > 3:
        return LIDPrediction("degenerate", 0.0)
    hits = sum(1 for w in words if w in lexicon)
    score = hits / len(words)
    return LIDPrediction(target if score >= 0.10 and hits >= 3 else "other", score)


def lid_supports(lid: LanguageIdentifier, target: str) -> bool:
    """Does the fastText backend carry a label for `target`?"""
    if getattr(lid, "backend", None) != "fasttext" or lid.model is None:
        return True  # non-fasttext backends decide in predict()
    labels = {l.replace("__label__", "") for l in lid.model.get_labels()}
    return target in labels


@dataclass
class LIDPrediction:
    label: str
    score: float


class LanguageIdentifier:
    def __init__(self, backend: str = "fasttext", model_path: str | None = None):
        self.backend = backend
        self.model = None
        if backend == "fasttext":
            if not model_path:
                raise ValueError("fasttext backend requires --lid-model path to lid.176.bin")
            import fasttext  # type: ignore

            self.model = fasttext.load_model(model_path)
        elif backend == "langdetect":
            from langdetect import DetectorFactory  # type: ignore

            DetectorFactory.seed = 0
        elif backend not in (None, "none"):
            raise ValueError(f"unknown LID backend: {backend}")

    def predict(self, text: str) -> LIDPrediction:
        if not text.strip():
            return LIDPrediction("", 0.0)
        if self.backend == "fasttext":
            labels, scores = self.model.predict(text.replace("\n", " "), k=1)
            label = labels[0].replace("__label__", "")
            return LIDPrediction(label, float(scores[0]))
        if self.backend == "langdetect":
            from langdetect import detect_langs  # type: ignore

            preds = detect_langs(text)
            if not preds:
                return LIDPrediction("", 0.0)
            return LIDPrediction(preds[0].lang, float(preds[0].prob))
        return LIDPrediction("", 0.0)


def compute_lss(outputs: list[str], target_language: str, lid: LanguageIdentifier) -> dict[str, Any]:
    target = canonical_lang(target_language)
    use_lexicon = not lid_supports(lid, target)
    if use_lexicon and target not in _LEXICONS:
        raise ValueError(
            f"LID backend has no label for {target} and no lexicon fallback; "
            "install a broader LID (e.g. GlotLID) or add a lexicon"
        )
    hits = 0
    details = []
    for text in outputs:
        pred = lexicon_lid_predict(text, target) if use_lexicon else lid.predict(text)
        ok = pred.label == target
        hits += int(ok)
        details.append({"pred": pred.label, "score": pred.score, "correct": ok})
    return {
        "target": target,
        "n": len(outputs),
        "lss": hits / max(len(outputs), 1),
        "lid_backend": "lexicon" if use_lexicon else getattr(lid, "backend", None),
        "details": details,
    }


def generate_plain(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 8,
    max_new_tokens: int = 48,
) -> list[str]:
    device = next(model.parameters()).device
    outs: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1] :]
            outs.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
    return outs
