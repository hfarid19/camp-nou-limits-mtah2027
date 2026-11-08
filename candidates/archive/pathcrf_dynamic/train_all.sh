#!/bin/sh
# Train the CRF: 7 leave-one-game-out folds (two processes at a time, 1 thread each), then the
# epoch choice from the mean held-out NLL curve and the full model.
#   sh candidates/pathcrf_dynamic/train_all.sh <python> <data dir> <out dir> [extra train_crf args]
PY=$1; DATA=$2; OUT=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/folds"
export OMP_NUM_THREADS=1
for pair in "0 1" "2 3" "4 5" "6"; do
  for gi in $pair; do
    "$PY" -I "$HERE/train_crf.py" --data "$DATA" --folds-out "$OUT/folds" --out "$OUT/crf.npz" --fold $gi "$@" > "$OUT/fold_$gi.log" 2>&1 &
  done
  wait
done
"$PY" -I "$HERE/train_crf.py" --data "$DATA" --folds-out "$OUT/folds" --out "$OUT/crf.npz" "$@"
