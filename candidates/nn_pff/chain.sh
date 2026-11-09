#!/bin/zsh
# One config's live checks as soon as its models exist:
#   candidates/nn_pff/chain.sh <config name> mstcn|tcn
# 1. PFF out-of-fold (run_pff.py, fold k's 12 games once fold_pff_<k>.npz exists),
# 2. when full.npz exists: folds JSON = ensemble_ctrl_v5c/folds.json + sha(full) -> [config dir, "tcn"],
#    then bench.py --dfl2 (leave-one-DFL-game-out fold models) -> probs/nn_pff_dfl/<name>.
set -e
NAME=$1; KIND=$2
R=/path/to/more-than-a-hack-2027
P=$MTAH_WORK/venv/bin/python
D=$MTAH_WORK/arch/nn_pff/$NAME
[ $KIND = mstcn ] && PIPE=$R/candidates/nn_pff/pipeline.py || PIPE=$R/candidates/nn_pff_tcn/pipeline.py
for k in 0 1 2 3; do
  while [ ! -f $D/fold_pff_$k.npz ]; do sleep 20; done
  [ $k = 0 ] && $P $R/candidates/nn_pff/folds_json.py $D
  G=$($P -c "import json;print(' '.join(json.load(open('$MTAH_WORK/arch/nn_pff/selection_folds.json'))['folds'][$k]))")
  $P $R/candidates/nn_pff/run_pff.py $PIPE --name $NAME --folds $D/pff_folds.json --model $D/fold_pff_0.npz --games ${=G}
done
while [ ! -f $D/full.npz ]; do sleep 20; done
$P - <<PY
import hashlib, json
f = json.load(open("$MTAH_WORK/arch/ensemble_ctrl_v5c/folds.json"))
f[hashlib.sha256(open("$D/full.npz", "rb").read()).hexdigest()[:12]] = ["$D", "tcn"]
json.dump(f, open("$D/folds_dfl.json", "w"), indent=1)
PY
cd $R && NN_PFF_MODEL=$D/full.npz $P candidates/bench.py ${PIPE:h}/pipeline_cfg.py --dfl2 --folds $D/folds_dfl.json \
    --save-probs $MTAH_WORK/probs/nn_pff_dfl/$NAME > $D/bench_dfl2.log 2>&1
tail -1 $D/bench_dfl2.log | cut -c1-200
