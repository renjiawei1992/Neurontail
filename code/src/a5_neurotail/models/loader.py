"""HuggingFace model/tokenizer loading."""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_cfg: dict[str, Any], device: str | None = None):
    hf_id = model_cfg["hf_id"]
    dtype = _DTYPES.get(model_cfg.get("dtype", "bfloat16"), torch.bfloat16)
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    if model_cfg.get("device_map"):
        kwargs["device_map"] = model_cfg["device_map"]
    if device:
        kwargs["device_map"] = device
    attn = model_cfg.get("attn_implementation")
    if attn:
        kwargs["attn_implementation"] = attn
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, trust_remote_code=bool(model_cfg.get("trust_remote_code", False))
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model.eval()
    return model, tokenizer


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def cuda_available() -> bool:
    return torch.cuda.is_available()


def describe_environment() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
