#!/bin/zsh
# Out-of-fold PFF scoring of one config as its fold models appear:
#   candidates/nn_pff/oof_chain.sh <config dir> <pipeline.py> <name>
# waits for fold_pff_<k>.npz (k = 0..3) and runs run_pff.py on fold k's 12 games.
set -e
D=$1; PIPE=$2; NAME=$3
R=${0:A:h:h:h}  # repo root
P=$MTAH_WORK/venv/bin/python
for k in 0 1 2 3; do
  while [ ! -f $D/fold_pff_$k.npz ]; do sleep 30; done
  if [ $k = 0 ]; then $P $R/candidates/nn_pff/folds_json.py $D; fi
  G=$($P -c "import json;print(' '.join(json.load(open('$MTAH_WORK/arch/nn_pff/selection_folds.json'))['folds'][$k]))")
  $P $R/candidates/nn_pff/run_pff.py $PIPE --name $NAME --folds $D/pff_folds.json --model $D/fold_pff_0.npz --games ${=G}
done
