#!/usr/bin/env bash
# Download the public evaluation data used by the scaffold.
# Usage: bash scripts/download_data.sh /data/a5_neurotail/raw
set -euo pipefail
DATA_DIR="${1:-./data/raw}"
mkdir -p "$DATA_DIR/belebele" "$DATA_DIR/flores"

echo "[1/2] Belebele"
curl -L --retry 3 -o "$DATA_DIR/belebele/Belebele.zip" \
  https://dl.fbaipublicfiles.com/belebele/Belebele.zip
unzip -q -o "$DATA_DIR/belebele/Belebele.zip" -d "$DATA_DIR/belebele/extracted"

echo "[2/2] FLORES-200"
curl -L --retry 3 -o "$DATA_DIR/flores/flores200_dataset.tar.gz" \
  https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
tar -xzf "$DATA_DIR/flores/flores200_dataset.tar.gz" -C "$DATA_DIR/flores"

cat <<'EOF'

Done. Manual steps:
  Glot500-c:   huggingface-cli download cis-lmu/Glot500 --repo-type dataset --local-dir data/raw/glot500
  Glot500 gated (religious / glotlid): request access, then download cis-lmu/glotlid-corpus
  OPUS sample: use https://opus.nlpl.eu/opusapi?source=en&target=xx (do not pull full NLLB)
  LID model:   download lid.176.bin from fastText and place at data/raw/lid.176.bin
EOF
