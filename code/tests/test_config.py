from pathlib import Path

from a5_neurotail.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_config_and_interpolation():
    cfg = load_config(
        ROOT / "configs/experiments.yaml",
        ROOT / "configs/languages.yaml",
        ROOT / "configs/models.yaml",
    )
    assert "llama31_8b" in cfg["models"]
    assert len(cfg["languages"]) >= 20
    assert "${" not in cfg["paths"]["processed_dir"]
    assert cfg["language_map"]["ton_Latn"]["tier"] == "ELRL"
