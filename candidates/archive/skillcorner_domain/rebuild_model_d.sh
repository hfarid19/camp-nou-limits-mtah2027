#!/bin/sh
# Rebuild model_d.txt (DFL + SkillCorner) and its 7 DFL leave-one-game-out fold models.
#   sh candidates/skillcorner_domain/rebuild_model_d.sh WORK_DIR DFL_R2_CACHE
# DFL_R2_CACHE is learned_dfl_evidence's round-2 cache (its ds/ made by
#   python -I candidates/learned_dfl_evidence/make_dataset.py --cache DFL_R2_CACHE ;
# see learned_dfl_evidence/rebuild_models.sh for fetch_idsse.py). model_ha.txt / model_c.txt are
# learned_dfl_evidence's files, copied byte-identically. Training uses 2 single-thread processes
# (~14 min folds + ~4 min full model on one laptop).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python}
W=$1; DFL=$2
mkdir -p "$W"
OMP_NUM_THREADS=1 "$PY" -I "$HERE/fetch_skillcorner.py" --out "$W/raw"
"$PY" -I "$HERE/parse_skillcorner.py" --raw "$W/raw" --out "$W/npz"
OMP_NUM_THREADS=1 "$PY" -I "$HERE/make_dataset_sc.py" --npz "$W/npz" --out "$W/ds_ex" --seeds 21 --players ex --jobs 2
OMP_NUM_THREADS=1 "$PY" -I "$HERE/train_sc.py" --dfl "$DFL/ds" --sc "$W/ds_ex" --models d --out "$W/m_v2" --folds "$W/m_v2/folds" --max-rounds 700 --jobs 2
cmp "$W/m_v2/model_d.txt" "$HERE/model_d.txt" && echo "model_d.txt: identical to the committed file"
