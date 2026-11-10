#!/bin/zsh
# Live checks of one nn_tsfm config as its models appear:
#   candidates/nn_tsfm/chain.sh <config name> [pff|dfl|both]
# pff:  PFF out-of-fold (nn_pff/run_pff.py on fold k's 12 games once fold_pff_<k>.npz exists)
#       -> $MTAH_WORK/probs/nn_pff/tsfm_<config>
# dfl:  when full.npz exists: folds JSON = ensemble_ctrl_v5c/folds.json + sha(full) -> [config dir, "tsfm"],
#       bench.py --dfl2 (leave-one-DFL-game-out fold_tsfm_<g>.npz) -> probs/nn_pff_dfl/tsfm_<config>
# Backend: NN_TSFM_BACKEND (numpy default; torch + NN_TSFM_DEVICE=mps for quick experiments).
set -e
NAME=$1; WHAT=${2:-both}
R=${0:A:h:h:h}  # repo root (this script is candidates/<family>/<name>.sh)
P=$MTAH_WORK/venv/bin/python
D=$MTAH_WORK/arch/nn_tsfm/$NAME
PIPE=$R/candidates/nn_tsfm/pipeline.py
if [ $WHAT != dfl ]; then
  for k in 0 1 2 3; do
    while [ ! -f $D/fold_pff_$k.npz ]; do sleep 30; done
    [ $k = 0 ] && $P $R/candidates/nn_pff/folds_json.py $D
    G=$($P -c "import json;print(' '.join(json.load(open('$MTAH_WORK/arch/nn_pff/selection_folds.json'))['folds'][$k]))")
    $P $R/candidates/nn_pff/run_pff.py $PIPE --name tsfm_$NAME --folds $D/pff_folds.json --model $D/fold_pff_0.npz --games ${=G}
  done
fi
if [ $WHAT != pff ]; then
  while [ ! -f $D/full.npz ]; do sleep 30; done
  for g in 0 1 2 3 4 5 6; do while [ ! -f $D/fold_tsfm_$g.npz ]; do sleep 30; done; done
  $P - <<PY
import hashlib, json
f = json.load(open("$MTAH_WORK/arch/ensemble_ctrl_v5c/folds.json"))
f[hashlib.sha256(open("$D/full.npz", "rb").read()).hexdigest()[:12]] = ["$D", "tsfm"]
json.dump(f, open("$D/folds_dfl.json", "w"), indent=1)
PY
  cd $R && NN_TSFM_MODEL=$D/full.npz $P candidates/bench.py candidates/nn_tsfm/pipeline_cfg.py --dfl2 --folds $D/folds_dfl.json \
      --save-probs $MTAH_WORK/probs/nn_pff_dfl/tsfm_$NAME > $D/bench_dfl2.log 2>&1
  tail -1 $D/bench_dfl2.log | cut -c1-200
fi
