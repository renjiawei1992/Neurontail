#!/usr/bin/env bash
# Aya-Expanse-8B full replication chain on GPU0 (GPU1 reserved for user's training).
set -euo pipefail
cd /home/ubuntu/home/python_project/A5-NeuronTail/code
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
OUT=/home/ubuntu/home/python_project/A5-NeuronTail/outputs
PY=.venv/bin/python

echo "=== [1/5] identify neurons (CPU) ==="
$PY scripts/identify_neurons.py --model aya_expanse_8b \
  --stats-dir $OUT/activations/aya_expanse_8b \
  --out-dir $OUT/neurons/aya_expanse_8b \
  --methods lape,dln,langfir --layers 12,16,20,24,28

echo "=== [2/5] baselines (7 langs) ==="
$PY - <<'EOF'
import sys, json, os
sys.path.insert(0, 'src')
from a5_neurotail.config import load_config
from a5_neurotail.data.schema import read_jsonl
from a5_neurotail.models.loader import load_model, set_seed
from a5_neurotail.eval.lss import generate_plain
cfg = load_config('configs/experiments.yaml', 'configs/languages.yaml', 'configs/models.yaml', 'configs/paths.yaml')
set_seed(0)
model, tokenizer = load_model(cfg['models']['aya_expanse_8b'])
base_dir = '/home/ubuntu/home/python_project/A5-NeuronTail/outputs/interventions/m2/aya_expanse_8b'
os.makedirs(base_dir, exist_ok=True)
for lang in ['swh_Latn','zul_Latn','amh_Ethi','bod_Tibt','fij_Latn','mri_Latn','eng_Latn']:
    prompts = [r['prompt'] for r in read_jsonl('/home/ubuntu/home/python_project/A5-NeuronTail/data/processed/prompts/lss_%s.jsonl' % lang)][:100]
    outs = generate_plain(model, tokenizer, prompts, max_new_tokens=48)
    with open(f'{base_dir}/{lang}_base.jsonl','w') as f:
        for p,o in zip(prompts,outs):
            f.write(json.dumps({'prompt':p,'output':o,'language':lang,'model':'aya_expanse_8b','key':'L24','method':'none'}, ensure_ascii=False)+'\n')
    print(lang, 'base done', flush=True)
del model
EOF

echo "=== [3/5] layer sweep fij/amh/mri x L16/20/24/28 (lape + 5 rand) ==="
$PY scripts/b2_layer_sweep.py --model aya_expanse_8b \
  --langs fij_Latn,amh_Ethi,mri_Latn --layers 16,20,24,28 --n-random 5

echo "=== [4/5] steering 6 long-tail x L24 (lape + 10 rand) ==="
$PY - <<'EOF'
import sys, json
import numpy as np
sys.path.insert(0, 'src')
from a5_neurotail.config import load_config
from a5_neurotail.data.schema import read_jsonl
from a5_neurotail.models.loader import load_model, set_seed
from a5_neurotail.models.modules import find_mlp_refs
from a5_neurotail.intervene.steering import steer_generate
cfg = load_config('configs/experiments.yaml', 'configs/languages.yaml', 'configs/models.yaml', 'configs/paths.yaml')
set_seed(0)
model, tokenizer = load_model(cfg['models']['aya_expanse_8b'])
refs = find_mlp_refs(model)
ref_map = {r.key: r for r in refs}
base_dir = '/home/ubuntu/home/python_project/A5-NeuronTail/outputs/interventions/m2/aya_expanse_8b'
sel_dir = '/home/ubuntu/home/python_project/A5-NeuronTail/outputs/neurons/aya_expanse_8b/lape'
for lang in ['swh_Latn','zul_Latn','amh_Ethi','bod_Tibt','fij_Latn','mri_Latn']:
    prompts = [r['prompt'] for r in read_jsonl('/home/ubuntu/home/python_project/A5-NeuronTail/data/processed/prompts/lss_%s.jsonl' % lang)][:100]
    sel = json.load(open(f'{sel_dir}/{lang}/L24.json'))['selected']
    outs = steer_generate(model, tokenizer, prompts, refs, 'L24', sel, 3.0, max_new_tokens=48)
    with open(f'{base_dir}/{lang}_steer_lape.jsonl','w') as f:
        for p,o in zip(prompts,outs):
            f.write(json.dumps({'prompt':p,'output':o,'language':lang,'model':'aya_expanse_8b','key':'L24','method':'steering','variant':'lape','alpha':3.0}, ensure_ascii=False)+'\n')
    for s in range(10):
        rng = np.random.default_rng(7000 + s)
        rand = np.sort(rng.choice(ref_map['L24'].intermediate_size, size=len(sel), replace=False))
        outs = steer_generate(model, tokenizer, prompts, refs, 'L24', rand, 3.0, max_new_tokens=48)
        with open(f'{base_dir}/{lang}_steer_rand{s:02d}.jsonl','w') as f:
            for p,o in zip(prompts,outs):
                f.write(json.dumps({'prompt':p,'output':o,'language':lang,'model':'aya_expanse_8b','key':'L24','method':'steering','variant':'random','seed':s}, ensure_ascii=False)+'\n')
    print(lang, 'steering done', flush=True)
del model
EOF

echo "=== [5/5] task domain eng/amh (knockout + steering, 3 rand) ==="
$PY scripts/m2_task_ko.py --model aya_expanse_8b --langs eng_Latn,amh_Ethi --n-random 3
$PY scripts/m2_task_steer.py --model aya_expanse_8b --langs eng_Latn,amh_Ethi --n-random 3

echo "=== Aya replication chain COMPLETE ==="
