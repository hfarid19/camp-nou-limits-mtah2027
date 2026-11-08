#!/bin/sh
# One-command rebuild of the three LightGBM models from the open DFL / IDSSE data, ending with a
# byte comparison against the committed model files.
#
#   sh candidates/learned_dfl_evidence/rebuild_models.sh [OUT_DIR]
#
# Needs network access (figshare article 28196177 v1, 21 files, ~2.6 GB download, md5-checked
# against idsse_manifest.json) and the pinned requirements.txt. Rough times on one laptop
# (18 cores): fetch + parse ~10-20 min (network bound), each make_dataset ~1 min, train ha
# ~4 min, train d,c ~3 min. Cache: $IDSSE_CACHE, else ./external/idsse (git-ignored).
#
# model_ha is trained on the round-1 dataset (no extra dropouts, no blackouts); model_d and
# model_c on the round-2 dataset (make_dataset.py defaults). See README.md.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python}
CACHE=${IDSSE_CACHE:-$HERE/external/idsse}
OUT=${1:-$CACHE/rebuilt_models}
mkdir -p "$CACHE/r1" "$CACHE/r2" "$OUT"
"$PY" -I "$HERE/fetch_idsse.py" --games 7 --cache "$CACHE"
ln -sfn "$CACHE/npz" "$CACHE/r1/npz"
ln -sfn "$CACHE/npz" "$CACHE/r2/npz"
"$PY" -I "$HERE/make_dataset.py" --cache "$CACHE/r1" --drop-p 0 --blackout-p 0
"$PY" -I "$HERE/make_dataset.py" --cache "$CACHE/r2"
rm -f "$OUT/model_meta.json"
"$PY" -I "$HERE/train.py" --cache "$CACHE/r1" --out "$OUT" --models ha
"$PY" -I "$HERE/train.py" --cache "$CACHE/r2" --out "$OUT" --models d,c
for f in model_d.txt model_ha.txt model_c.txt model_meta.json; do
    cmp "$OUT/$f" "$HERE/$f" && echo "$f: identical to the committed file"
done
