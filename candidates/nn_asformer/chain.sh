#!/bin/zsh
# One config's live checks as its models appear (= candidates/nn_pff/chain.sh for nn_asformer):
#   candidates/nn_asformer/chain.sh <config name> [pipeline.py]
# 1. PFF out-of-fold (run_pff.py; fold k's 12 games once fold_pff_<k>.npz exists) -> probs/nn_asformer/<name>,
# 2. when full.npz exists: folds JSON = ensemble_ctrl_v5c/folds.json + sha(full) -> [config dir, "tcn"],
#    then bench.py --dfl2 (leave-one-DFL-game-out fold models) -> probs/nn_asformer_dfl/<name>.
set -e
NAME=$1
R=${0:A:h:h:h}  # repo root (this script is candidates/<family>/<name>.sh)
P=$MTAH_WORK/venv/bin/python
D=$MTAH_WORK/arch/nn_asformer/$NAME
PIPE=${2:-$R/candidates/nn_asformer/pipeline.py}
for k in 0 1 2 3; do
  while [ ! -f $D/fold_pff_$k.npz ]; do sleep 30; done
  [ $k = 0 ] && $P $R/candidates/nn_pff/folds_json.py $D
  G=$($P -c "import json;print(' '.join(json.load(open('$MTAH_WORK/arch/nn_pff/selection_folds.json'))['folds'][$k]))")
  $P $R/candidates/nn_asformer/run_pff.py $PIPE --name $NAME --folds $D/pff_folds.json --model $D/fold_pff_0.npz --games ${=G}
done
[ "$3" = "pff-only" ] && exit 0
while [ ! -f $D/full.npz ]; do sleep 30; done
for g in 0 1 2 3 4 5 6; do while [ ! -f $D/fold_tcn_$g.npz ]; do sleep 30; done; done
$P - <<PY
import hashlib, json
f = json.load(open("$MTAH_WORK/arch/ensemble_ctrl_v5c/folds.json"))
f[hashlib.sha256(open("$D/full.npz", "rb").read()).hexdigest()[:12]] = ["$D", "tcn"]
json.dump(f, open("$D/folds_dfl.json", "w"), indent=1)
PY
cd $R && NN_ASF_MODEL=$D/full.npz OMP_NUM_THREADS=2 $P candidates/bench.py candidates/nn_asformer/pipeline_cfg.py --dfl2 --folds $D/folds_dfl.json \
    --save-probs $MTAH_WORK/probs/nn_asformer_dfl/$NAME > $D/bench_dfl2.log 2>&1
tail -1 $D/bench_dfl2.log | cut -c1-300
