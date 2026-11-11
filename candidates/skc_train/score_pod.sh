#!/bin/bash
# Round-5 "+SKC" scoring on the training pod (bash), one variant at a time, every game in its own process:
#   candidates/skc_train/score_pod.sh <variant dir> ptr|mstcn <name> skccv|nowin|dfl|all
# skccv  SkillCorner cross-validation: game g of fold k is run ONLY with <dir>/fold_skc_<k>.npz (asserted: its
#        train_games do not contain g) -> probs/skc/<name>, probs/skc_mbr/<name> (nn_pointer/skc_both.py layout)
# nowin  whole-game run of <dir>/nowin.npz on all 20 games -> probs/skc/<name>_nowin, probs/skc_mbr/<name>_nowin
#        (scored on the 10-minute windows only, on the Mac: skc_train/score.py)
# dfl    bench.py --dfl2 with the 7 LOGO models (folds JSON = ensemble_ctrl_v5c/folds.json + sha(full) -> dir),
#        one process per DFL game (BENCH_DFL_ONLY) -> probs/nn_pff_dfl/<name>
# Python: $PY (default /root/skcscore/bin/python: numpy 2.5.4 / pandas 3.1.1 / scipy 1.18.1 / lightgbm 4.7.0 = the Mac).
set -u
D=$1; KIND=$2; NAME=$3; WHAT=$4
R=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-/root/skcscore/bin/python}
export MTAH_WORK=${MTAH_WORK:-/root/work} OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} NN_PFF_SKC_CV=1
case $KIND in mstcn) PIPE=$R/candidates/nn_pff/pipeline_cfg.py;; ptr) PIPE=$R/candidates/nn_pointer/pipeline_cfg.py;; esac
cd $R
LOG=$D/score_logs; mkdir -p $LOG
if [ $WHAT = skccv ] || [ $WHAT = all ]; then
  for k in 0 1 2 3; do
    G=$($PY -I - <<PY
import sys; sys.path.insert(0, "candidates/nn_pff")
import numpy as np, data as D
games = D.skc_cv_folds()[$k]
tg = {str(g) for g in np.load("$D/fold_skc_$k.npz")["train_games"]}
assert not tg & set(games), ("fold model trained on a scored game", tg & set(games))
assert len([g for g in tg if g.startswith("SKC-")]) == 15, "expected the other 15 SKC games"
print(" ".join(g[4:] for g in games))
PY
)
    [ -z "$G" ] && { echo "fold $k check FAILED"; exit 1; }
    for g in $G; do
      NN_PFF_MODEL=$D/fold_skc_$k.npz $PY -I candidates/nn_pointer/skc_both.py $PIPE $NAME --games $g > $LOG/skccv_$g.log 2>&1 &
    done
  done
fi
if [ $WHAT = nowin ] || [ $WHAT = all ]; then
  for g in $(ls $MTAH_WORK/skc_aleague/npz | sed 's/SKC-//; s/.npz//'); do
    NN_PFF_MODEL=$D/nowin.npz $PY -I candidates/nn_pointer/skc_both.py $PIPE ${NAME}_nowin --games $g > $LOG/nowin_$g.log 2>&1 &
  done
fi
if [ $WHAT = dfl ] || [ $WHAT = all ]; then
  $PY -I - <<PY
import hashlib, json
f = json.load(open("$MTAH_WORK/arch/ensemble_ctrl_v5c/folds.json"))
f[hashlib.sha256(open("$D/full.npz", "rb").read()).hexdigest()[:12]] = ["$D", "tcn"]
json.dump(f, open("$D/folds_dfl.json", "w"), indent=1)
PY
  for m in $(ls $MTAH_WORK/idsse/npz | sed 's/.npz//'); do
    BENCH_DFL_ONLY=$m NN_PFF_MODEL=$D/full.npz $PY -I candidates/bench.py $PIPE --dfl2 --folds $D/folds_dfl.json \
        --save-probs $MTAH_WORK/probs/nn_pff_dfl/$NAME > $LOG/dfl_$m.log 2>&1 &
  done
fi
wait
echo "SCORE_DONE $NAME $WHAT"
