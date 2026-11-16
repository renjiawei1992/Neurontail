"""Multiple-choice scoring (Belebele / native probes)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..data.prompts import mcqa_prompt


def _option_logprob(model, tokenizer, prompt: str, option: str, max_length: int) -> float:
    device = next(model.parameters()).device
    prefix = tokenizer(prompt, add_special_tokens=False).input_ids
    full = tokenizer(prompt + " " + option, add_special_tokens=False).input_ids
    if len(full) <= len(prefix):
        return float("-inf")
    ids = torch.tensor([full[-max_length:]], device=device)
    prefix_len = min(len(prefix), ids.shape[1] - 1)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits[:, :-1, :]
        logprobs = F.log_softmax(logits, dim=-1)
        target = ids[:, 1:]
        take = logprobs[0, prefix_len:, :].gather(
            1, target[0, prefix_len:].unsqueeze(-1)
        ).squeeze(-1)
    return float(take.mean().item())


def score_options(
    model, tokenizer, prompt: str, options: list[str], max_length: int = 512
) -> list[float]:
    return [_option_logprob(model, tokenizer, prompt, opt, max_length) for opt in options]


def evaluate_mcqa(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    max_length: int = 512,
    limit: int | None = None,
    ctx=None,
) -> dict[str, Any]:
    """Score MCQA; pass `ctx` (a context manager installing intervention
    hooks) to measure accuracy under neuron interventions."""
    from contextlib import nullcontext

    rows = []
    correct = 0
    items = examples[:limit] if limit else examples
    with ctx if ctx is not None else nullcontext():
        for ex in items:
            prompt = mcqa_prompt(ex["passage"], ex["question"], ex["options"])
            scores = score_options(model, tokenizer, prompt, ex["options"], max_length)
            pred = int(max(range(len(scores)), key=lambda i: scores[i]))
            ok = pred == int(ex["answer_idx"])
            correct += int(ok)
            rows.append(
                {
                    "id": ex.get("id"),
                    "language": ex.get("language"),
                    "pred": pred,
                    "answer": int(ex["answer_idx"]),
                    "correct": ok,
                    "scores": scores,
                }
            )
    return {"n": len(rows), "accuracy": correct / max(len(rows), 1), "rows": rows}
