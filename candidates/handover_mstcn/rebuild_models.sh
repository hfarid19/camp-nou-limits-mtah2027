#!/bin/sh
# Rebuild model_tcn.npz from the open DFL / IDSSE data (see README.md).
#   sh candidates/handover_mstcn/rebuild_models.sh CACHE_DIR OUT_DIR
# CACHE_DIR: an IDSSE cache made by learned_dfl_evidence/fetch_idsse.py (CACHE_DIR/npz).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
LDE="$HERE/../learned_dfl_evidence"
PY=${PYTHON:-python}
CACHE=$1
OUT=$2
mkdir -p "$CACHE/r2" "$OUT"
ln -sfn "$CACHE/npz" "$CACHE/r2/npz"
"$PY" -I "$LDE/make_dataset.py" --cache "$CACHE/r2"                      # round-2 settings, seeds 11-13
"$PY" -I "$HERE/prep_data.py" --ds "$CACHE/r2/ds" --npz "$CACHE/npz" --out "$OUT/data"
OMP_NUM_THREADS=2 "$PY" -I "$HERE/train_tcn.py" --data "$OUT/data" --out "$OUT/folds" \
    --folds 0,1,2,3,4,5,6 --threads 2 --cfg '{"epochs": 15}'
"$PY" -I "$HERE/pack_model.py" "$OUT/folds" "$OUT/folds.json"
