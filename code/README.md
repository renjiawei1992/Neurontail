# A5 NeuronTail

Code scaffold for **language neurons, causal specificity, and steering in low-resource / extremely low-resource languages**.

This repository is built to be uploaded to a GPU server and run there. It does **not** ship model weights or run models during setup.

> **Data path note**: this folder is `A5-NeuronTail/code/`; in the commands below, `data/` means the project-level `../data/`. Adjust `configs/paths.yaml` accordingly.

---

## What the code supports

- **Data**: FLORES-200, Belebele, native CSV probes, OPUS-style TSV; unified JSONL schemas.
- **Activation statistics**: streaming per-neuron `count / positive count / mean / histogram` for dense MLPs and MoE experts (Llama/Qwen/Gemma/Mistral/OLMoE-style `gate_proj/up_proj/down_proj`).
- **Identification**: LAPE, distribution-aware (DLN), LangFIR random-token control, CRANE-style causal relevance.
- **Interventions**: neuron knockout, activation addition/steering, donor patch, romanization/script control.
- **Evaluation**: MCQA log-likelihood scoring, LSS with a pluggable language ID backend, causal specificity gap (CSG), donor transfer matrix.
- **Training**: full LoRA, random-subnetwork LoRA, neuron-targeted LoRA (gradient-masked approximation).
- **Analysis**: bootstrap CI, mixed-effects models, change-point, paper tables.

---

## Quick start (server)

```bash
cd a5_neurotail
python -m venv .venv && source .venv/bin/activate
# install the torch build that matches your CUDA first, e.g.
# pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
# optional: LSS + romanization
pip install fasttext-wheel langdetect PyICU
```

Set paths (or copy `configs/paths.example.yaml` and pass `--paths`):

```bash
cp configs/paths.example.yaml configs/paths.yaml
# edit configs/paths.yaml
```

HuggingFace login for gated models:

```bash
huggingface-cli login
```

### 1. Prepare data

```bash
python scripts/prepare_data.py \
  --config configs/experiments.yaml \
  --flores-dir data/raw/flores \
  --belebele-dir data/raw/belebele/extracted/Belebele \
  --native-dir data/processed/native \
  --main-only
```

Outputs: `data/processed/parallel/{code}.jsonl`, `data/processed/mcqa/*.jsonl`, `data/processed/prompts/lss_{code}.jsonl`.

### 2. Extract activation statistics

```bash
python scripts/extract_activations.py \
  --model llama31_8b \
  --main-only \
  --max-texts 1000
```

Outputs: `outputs/activations/{model}/{code}.npz` + `.json`, plus `_random.npz` for LangFIR.

### 3. Identify neurons

```bash
python scripts/identify_neurons.py \
  --model llama31_8b \
  --methods lape,dln,langfir \
  --layers 8,12,16,20,24,28
```

Outputs: `outputs/neurons/{model}/{method}/{code}/{key}.json` with `selected` indices.

### 4. Run an intervention

```bash
python scripts/run_interventions.py \
  --model llama31_8b \
  --selection outputs/neurons/llama31_8b/lape/mri_Latn/L12.json \
  --key L12 \
  --language mri_Latn \
  --method steering \
  --alpha 2.0 \
  --out outputs/interventions/llama31_8b/mri_Latn_L12_steer.jsonl
```

### 5. Evaluate

```bash
# LSS
python scripts/evaluate.py \
  --task lss \
  --data outputs/interventions/llama31_8b/mri_Latn_L12_steer.jsonl \
  --lid-model data/raw/lid.176.bin \
  --out outputs/eval/lss_mri_L12.json

# MCQA
python scripts/evaluate.py \
  --task mcqa \
  --model llama31_8b \
  --data data/processed/mcqa/belebele_mri_Latn.jsonl \
  --out outputs/eval/belebele_mri_L12.json
```

### 6. Neuron-targeted LoRA (optional)

```bash
python scripts/train_lora.py \
  --model llama31_8b \
  --data data/processed/parallel/mri_Latn.jsonl \
  --selection outputs/neurons/llama31_8b/lape/mri_Latn/L12.json \
  --key L12 \
  --mode neuron_lora \
  --out outputs/lora/llama31_8b/mri_neuron
```

---

## Pipeline

```
FLORES / Belebele / native CSV
        │
        ▼
prepare_data.py  →  processed/{parallel,mcqa,prompts}
        │
        ▼
extract_activations.py  →  activations/{model}/{lang}.npz
        │
        ▼
identify_neurons.py  →  neurons/{model}/{method}/{lang}/{key}.json
        │
        ├── run_interventions.py  →  interventions/*.jsonl
        │        │
        │        ▼
        │   evaluate.py (LSS / MCQA)
        │
        └── train_lora.py  →  lora/{model}/{lang}_{mode}
                 │
                 ▼
          evaluate.py + analyze_results.py + make_tables.py
```

---

## Repository layout

```
a5_neurotail/
├── configs/
│   ├── experiments.yaml     # paths, batch sizes, k/alpha, seeds
│   ├── languages.yaml       # 32 languages with tier/script/family/main flag
│   └── models.yaml          # 6 models
├── src/a5_neurotail/
│   ├── config.py            # ${key} interpolation
│   ├── data/                # schema, loaders, normalize, fertility, prompts
│   ├── models/              # loader, MLP discovery, hooks, activations
│   ├── identify/            # lape, dln, langfir, crane
│   ├── intervene/           # knockout, steering, patching, script
│   ├── eval/                # lss, mcqa, csg, transfer
│   ├── train/               # lora
│   └── analysis/            # stats, tables
├── scripts/                 # CLI entry points
├── tests/                   # pure-python tests (no model runs)
└── data/                    # raw/processed data (gitignored)
```

---

## Experimental mapping

| Block | Code path | Main outputs |
|---|---|---|
| **B1** construct validity | `extract_activations.py` + `identify_neurons.py` | LAPE/DLN/LangFIR selections, fertility |
| **B2** causal decomposition | `run_interventions.py` (knockout/steering) + `evaluate.py` | CSG, LSS, task accuracy |
| **B3** script×family donor | `run_interventions.py` + donor stats + `eval/transfer.py` | donor transfer matrix |
| **B4** resource gradient | `analyze_results.py` + `analysis/stats.py` | CSG vs resource / change-point |
| **B5** generalization | multi-model runs + `train_lora.py` | cross-model tables, adaptation gain |

---

## Key config knobs

`configs/experiments.yaml`:

- `activation.batch_size`, `max_length`, `token_position` (`all` / `last`), `hist_bins`.
- `identify.top_frac`, `min_prob`, `random_tokens`, `random_pos_threshold`.
- `interventions.k_fracs`, `n_random_sets`, `alphas`.
- `eval.lid_backend`, `lid_model`.
- `lora.r/alpha/lr/epochs`, `modes`.

`configs/languages.yaml`: `main: true` selects the 12-language main set; `tier` is high/mid/LRL/ELRL; `data_tier` is A/A-/B/C.

---

## Server notes

- **Models**: Llama-3.1, Gemma-2, Aya are gated; accept licenses and `huggingface-cli login`.
- **Memory**: activation extraction uses forward passes only; start with `--max-texts 500` and `batch_size 2`.
- **Single GPU for hooks**: use one GPU (`device_map: {"": 0}`) for activation extraction and interventions; multi-GPU sharding can put hooked modules and inputs on different devices.
- **Storage**: stats are small (`N` per layer, not full activations); a 32-layer 8B model is a few MB per language.
- **Attention implementation**: config uses `eager` for reliable hooks; switch to `sdpa`/`flash_attention_2` only after verifying hooks.
- **MoE**: OLMoE experts are discovered as `L{layer}E{expert}`; decide whether to pool experts or analyze them separately.
- **LSS**: download `lid.176.bin` from the fastText language identification page and put it at `data/raw/lid.176.bin`.
- **Romanization**: requires PyICU; otherwise pre-romanize text and use `intervene/script.py` only as a wrapper.

---

## Known limitations / TODO

1. **Tokenizer fertility** is not computed in this scaffold; add a `--with-tokenizer` step using the model tokenizer before B1.
2. **Neuron LoRA** is a gradient-masked approximation (gate/up rows, down columns), not a hard parameter subset. Validate the mask on a small model before scaling.
3. **MoE semantics**: `gate_proj` hooks capture expert-internal neurons; routing analysis needs a separate hook on the router logits.
4. **LAPE normalization** uses reference languages from the run; if you add/remove languages, recompute selections.
5. **CRANE** is O(#candidates) forward passes; cap with `max_neurons` and use it on a subset.
6. **LSS** depends on the fastText model's label set; map ISO codes carefully (see `eval/lss.py`).
7. **Data licenses**: FLORES/Belebele are CC-BY-SA; OPUS subsets vary; Glot500 has mixed licenses. Check before redistribution.

---

## Tests

```bash
pytest
```

The tests cover pure logic only (LAPE math, CSG, schema, config, fertility); they do not load models.
