# Neurontail

Code and results accompanying our paper on layer-localized causal necessity
of language neurons and the control--task dissociation in long-tail
languages.

## Repository structure

```
├── code/                          # experiment code (Python package + CLI scripts)
│   ├── src/a5_neurotail/          # data / models / identify / intervene / eval / train / analysis
│   ├── scripts/                   # pipeline entry points (prepare_data → extract → identify → intervene → evaluate)
│   ├── configs/                   # experiments.yaml, languages.yaml, models.yaml, paths.example.yaml
│   └── tests/
└── results/
    ├── neurons/{model}/lape/{lang}/L{layer}.json   # released neuron selections (top-1% indices)
    ├── interventions/{m2,b2sweep,b3}/{model}/*.jsonl  # all generated outputs behind every table/figure
    ├── eval/                      # scored results: per-seed LSS table (glotlid_all_v3.csv), task CSG/steering JSONs
    └── fertility/                 # tokenizer-fertility tables
```

## Reproducing the pipeline

```bash
# 1. environment (GPU, Python >= 3.10)
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt && pip install -e code/

# 2. data (FLORES-200 dev, Belebele; CC-BY-SA 4.0)
bash code/scripts/download_data.sh data/raw

# 3. per-language processed data
python code/scripts/prepare_data.py --paths code/configs/paths.yaml --main-only

# 4. activation statistics → identification → interventions → evaluation
python code/scripts/extract_activations.py --model qwen25_7b --paths code/configs/paths.yaml --main-only
python code/scripts/identify_neurons.py --model qwen25_7b --layers 8,12,16,20,24,28
# (knockout / steering / patching drivers: scripts/m2_causal_block.py,
#  scripts/b2_layer_sweep.py, scripts/b3_donor_matrix.py, scripts/m2_task_ko.py, ...)
```

All numbers in the paper's tables are recomputable from
`results/interventions/` with GlotLID (`cis-lmu/glotlid`).

## Licenses

- Code: MIT (see `code/LICENSE`).
- Intervention outputs derive from FLORES-200 and Belebele
  (both CC-BY-SA 4.0); retain attribution if redistributed.
- Models (Qwen2.5-7B-Instruct, Mistral-Nemo-12B-Instruct) and GlotLID are
  distributed under their respective licenses by their authors.
