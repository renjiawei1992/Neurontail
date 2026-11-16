"""Config loading with simple ${key} interpolation."""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

_VAR = re.compile(r"\$\{([^}]+)\}")


def _expand(value: Any, root: dict[str, Any], depth: int = 0) -> Any:
    if depth > 10:
        raise ValueError("config interpolation too deep")
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            key = m.group(1)
            if key in os.environ:
                return os.environ[key]
            # Bare keys such as ${data_root} resolve inside `paths` first.
            if "." not in key and isinstance(root.get("paths"), dict) and key in root["paths"]:
                return str(root["paths"][key])
            cur: Any = root
            for part in key.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return m.group(0)
                cur = cur[part]
            return str(cur)

        out = _VAR.sub(repl, value)
        return out
    if isinstance(value, list):
        return [_expand(v, root, depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, root, depth + 1) for k, v in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return data


def load_config(
    experiments: str | Path = "configs/experiments.yaml",
    languages: str | Path = "configs/languages.yaml",
    models: str | Path = "configs/models.yaml",
    paths: str | Path | None = None,
) -> dict[str, Any]:
    """Load and merge configs. `paths` overrides `experiments.paths`."""
    exp = load_yaml(experiments)
    lang = load_yaml(languages)
    mod = load_yaml(models)
    cfg: dict[str, Any] = {
        "project": exp.get("project", "a5_neurotail"),
        "seed": exp.get("seed", 0),
        "paths": copy.deepcopy(exp.get("paths", {})),
        "activation": exp.get("activation", {}),
        "identify": exp.get("identify", {}),
        "interventions": exp.get("interventions", {}),
        "eval": exp.get("eval", {}),
        "lora": exp.get("lora", {}),
        "analysis": exp.get("analysis", {}),
        "languages": lang.get("languages", []),
        "models": mod.get("models", {}),
    }
    if paths:
        cfg["paths"].update(load_yaml(paths))
    cfg = _expand(cfg, cfg)
    cfg["language_map"] = {x["code"]: x for x in cfg["languages"]}
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    p = Path(cfg["paths"][key]).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def languages_for_run(cfg: dict[str, Any], main_only: bool = False) -> list[dict[str, Any]]:
    langs = cfg["languages"]
    if main_only:
        langs = [x for x in langs if x.get("main")]
    return langs
